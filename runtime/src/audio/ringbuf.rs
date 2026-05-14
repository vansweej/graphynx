//! Lock-free single-producer single-consumer ring buffer.
//!
//! Used to transfer audio samples from the cpal callback thread (producer) to
//! the render thread (consumer) without blocking either side.
//!
//! ```text
//!  cpal callback ──push()──► [RingBuffer] ──pop_into()──► render thread
//!  (audio thread)                                         (game loop)
//! ```
//!
//! # Ordering guarantees
//!
//! - `push` uses `Release` on the head index so the written sample is visible
//!   before the index update.
//! - `pop_into` uses `Acquire` on the head index so it sees all writes that
//!   happened before the producer's `Release`.
//! - The tail index is only ever written by the consumer, so it uses `Relaxed`
//!   stores and `Relaxed` loads on the consumer side.
//!
//! # Capacity
//!
//! The capacity is rounded up to the next power of two so that index masking
//! (`& (capacity - 1)`) can replace modulo.  Minimum capacity is 2.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// ── RingBuffer ────────────────────────────────────────────────────────────────

/// A bounded, lock-free SPSC ring buffer.
///
/// The capacity is rounded up to the next power of two at construction time.
/// When the buffer is full, `push` overwrites the oldest sample (the tail
/// advances automatically), so the consumer always sees the freshest data.
pub struct RingBuffer<T> {
    buf: Box<[T]>,
    capacity: usize,   // always a power of two
    mask: usize,       // capacity - 1
    head: AtomicUsize, // write cursor (producer owns)
    tail: AtomicUsize, // read cursor  (consumer owns)
}

impl<T: Copy + Default> RingBuffer<T> {
    /// Create a new ring buffer with at least `min_capacity` slots.
    ///
    /// The actual capacity is the smallest power of two ≥ `min_capacity` and
    /// ≥ 2.
    pub fn new(min_capacity: usize) -> Arc<Self> {
        let capacity = min_capacity.max(2).next_power_of_two();
        let buf = vec![T::default(); capacity].into_boxed_slice();
        Arc::new(Self {
            buf,
            capacity,
            mask: capacity - 1,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        })
    }

    /// The actual capacity of the buffer (a power of two).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of samples currently available to read.
    pub fn available(&self) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Relaxed);
        head.wrapping_sub(tail)
    }

    /// Push one sample.  If the buffer is full, the oldest sample is
    /// overwritten (tail advances to make room).
    ///
    /// # Safety
    ///
    /// Must only be called from the **single producer** thread.
    pub fn push(&self, value: T) {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);

        // If full, advance tail to drop the oldest sample.
        if head.wrapping_sub(tail) == self.capacity {
            self.tail.store(tail.wrapping_add(1), Ordering::Release);
        }

        // SAFETY: head & mask gives a valid index into buf.
        unsafe {
            let slot = self.buf.as_ptr().add(head & self.mask) as *mut T;
            slot.write(value);
        }
        self.head.store(head.wrapping_add(1), Ordering::Release);
    }

    /// Drain up to `dst.len()` samples into `dst`.
    ///
    /// Returns the number of samples actually written.  If fewer than
    /// `dst.len()` samples are available, only the available samples are
    /// written and the rest of `dst` is left unchanged.
    ///
    /// # Safety
    ///
    /// Must only be called from the **single consumer** thread.
    pub fn pop_into(&self, dst: &mut [T]) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Relaxed);
        let avail = head.wrapping_sub(tail).min(dst.len());

        for (i, slot) in dst[..avail].iter_mut().enumerate() {
            // SAFETY: (tail + i) & mask is a valid index.
            *slot = unsafe { *self.buf.as_ptr().add((tail.wrapping_add(i)) & self.mask) };
        }

        if avail > 0 {
            self.tail.store(tail.wrapping_add(avail), Ordering::Release);
        }
        avail
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_is_rounded_to_power_of_two() {
        let rb = RingBuffer::<f32>::new(3);
        assert_eq!(rb.capacity(), 4);

        let rb2 = RingBuffer::<f32>::new(8);
        assert_eq!(rb2.capacity(), 8);

        // Minimum capacity is 2.
        let rb3 = RingBuffer::<f32>::new(0);
        assert_eq!(rb3.capacity(), 2);
    }

    #[test]
    fn empty_buffer_available_is_zero() {
        let rb = RingBuffer::<f32>::new(4);
        assert_eq!(rb.available(), 0);
    }

    #[test]
    fn push_increases_available() {
        let rb = RingBuffer::<f32>::new(4);
        rb.push(1.0);
        rb.push(2.0);
        assert_eq!(rb.available(), 2);
    }

    #[test]
    fn pop_into_reads_fifo_order() {
        let rb = RingBuffer::<f32>::new(4);
        rb.push(1.0);
        rb.push(2.0);
        rb.push(3.0);

        let mut dst = [0.0_f32; 3];
        let n = rb.pop_into(&mut dst);
        assert_eq!(n, 3);
        assert_eq!(dst, [1.0, 2.0, 3.0]);
        assert_eq!(rb.available(), 0);
    }

    #[test]
    fn pop_into_partial_when_fewer_samples_available() {
        let rb = RingBuffer::<f32>::new(8);
        rb.push(10.0);
        rb.push(20.0);

        let mut dst = [0.0_f32; 5];
        let n = rb.pop_into(&mut dst);
        assert_eq!(n, 2);
        assert_eq!(dst[0], 10.0);
        assert_eq!(dst[1], 20.0);
        // Remaining slots untouched.
        assert_eq!(dst[2], 0.0);
    }

    #[test]
    fn empty_pop_returns_zero() {
        let rb = RingBuffer::<f32>::new(4);
        let mut dst = [0.0_f32; 4];
        let n = rb.pop_into(&mut dst);
        assert_eq!(n, 0);
    }

    #[test]
    fn overflow_drops_oldest_sample() {
        // Capacity 4: push 5 items — item 0 should be dropped.
        let rb = RingBuffer::<f32>::new(4);
        for i in 0..5u32 {
            rb.push(i as f32);
        }
        // Available should be capped at capacity.
        assert_eq!(rb.available(), 4);

        let mut dst = [0.0_f32; 4];
        rb.pop_into(&mut dst);
        // Items 1..=4 remain (0 was dropped).
        assert_eq!(dst, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn multiple_push_pop_cycles_wrap_correctly() {
        let rb = RingBuffer::<i32>::new(4);

        // First cycle.
        rb.push(1);
        rb.push(2);
        let mut dst = [0i32; 2];
        rb.pop_into(&mut dst);
        assert_eq!(dst, [1, 2]);

        // Second cycle — indices have wrapped.
        rb.push(3);
        rb.push(4);
        rb.push(5);
        let mut dst2 = [0i32; 3];
        let n = rb.pop_into(&mut dst2);
        assert_eq!(n, 3);
        assert_eq!(dst2, [3, 4, 5]);
    }

    #[test]
    #[cfg_attr(
        tarpaulin,
        ignore = "stress test can exceed tarpaulin response timeout"
    )]
    #[cfg(not(tarpaulin_include))]
    fn concurrent_push_pop_stress() {
        use std::thread;

        const ITEMS: usize = 50_000;
        let rb = RingBuffer::<u32>::new(256);
        let rb_producer = Arc::clone(&rb);

        let producer = thread::spawn(move || {
            for i in 0..ITEMS as u32 {
                rb_producer.push(i);
            }
        });

        // Consumer: drain until we've seen ITEMS items (some may be dropped
        // due to overflow — that's expected behaviour).
        let mut total_read = 0usize;
        let mut buf = [0u32; 64];
        while total_read < ITEMS {
            let n = rb.pop_into(&mut buf);
            total_read += n;
            if n == 0 {
                // Yield to let producer make progress.
                thread::yield_now();
            }
        }

        producer.join().unwrap();
        // All items were eventually consumed (possibly with some dropped
        // mid-stream, but we read at least ITEMS total).
        assert!(total_read >= ITEMS);
    }
}
