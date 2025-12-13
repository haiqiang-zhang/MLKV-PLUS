import random
import math
import numpy as np
from collections import Counter

class ZipfianGenerator:
    ZIPFIAN_CONSTANT = 0.99

    def __init__(self, min_val=0, max_val=None, zipfian_constant=ZIPFIAN_CONSTANT, zetan=None):
        """
        Create a zipfian generator for items between min and max (inclusive).
        
        Args:
            min_val: The smallest integer to generate in the sequence
            max_val: The largest integer to generate in the sequence
            zipfian_constant: The zipfian constant to use
            zetan: Precomputed zeta constant (optional)
        """
        if max_val is None:
            max_val = min_val - 1
            min_val = 0

        self.items = max_val - min_val + 1
        self.base = min_val
        self.zipfian_constant = zipfian_constant
        self.theta = self.zipfian_constant
        self.allow_item_count_decrease = False
        self.count_for_zeta = self.items

        # Compute zeta values
        if zetan is None:
            self.zetan = self._zeta_static(self.items, self.theta)
        else:
            self.zetan = zetan

        self.zeta2theta = self._zeta_static(2, self.theta)
        self.alpha = 1.0 / (1.0 - self.theta)
        self.eta = (1 - math.pow(2.0 / self.items, 1 - self.theta)) / (1 - self.zeta2theta / self.zetan)

        self.next_value()

    def _zeta_static(self, n, theta, st=0, initial_sum=0):
        """
        Compute the zeta constant needed for the distribution.
        
        Args:
            n: The number of items to compute zeta over
            theta: The zipfian constant
            st: The number of items used to compute the last initial_sum
            initial_sum: The value of zeta we are computing incrementally from
        """
        sum_val = initial_sum
        for i in range(st, n):
            sum_val += 1 / (math.pow(i + 1, theta))
        return sum_val

    def _zeta(self, n, theta, st=0, initial_sum=0):
        """
        Compute the zeta constant needed for the distribution.
        Remember the new value of n so that if we change the itemcount,
        we'll know to recompute zeta.
        """
        self.count_for_zeta = n
        return self._zeta_static(n, theta, st, initial_sum)

    def next_long(self, item_count):
        """
        Generate the next item as a long.
        
        Args:
            item_count: The number of items in the distribution
        Returns:
            The next item in the sequence
        """
        if item_count != self.count_for_zeta:
            if item_count > self.count_for_zeta:
                # Incrementally compute zetan
                self.zetan = self._zeta(self.count_for_zeta, item_count, self.theta, self.zetan)
                self.eta = (1 - math.pow(2.0 / self.items, 1 - self.theta)) / (1 - self.zeta2theta / self.zetan)
            elif item_count < self.count_for_zeta and self.allow_item_count_decrease:
                # Recompute zetan from scratch
                print(f"WARNING: Recomputing Zipfian distribution. This is slow and should be avoided. "
                      f"(itemcount={item_count} countforzeta={self.count_for_zeta})")
                self.zetan = self._zeta(item_count, self.theta)
                self.eta = (1 - math.pow(2.0 / self.items, 1 - self.theta)) / (1 - self.zeta2theta / self.zetan)

        u = random.random()
        uz = u * self.zetan

        if uz < 1.0:
            return self.base
        if uz < 1.0 + math.pow(0.5, self.theta):
            return self.base + 1

        ret = self.base + int((item_count) * math.pow(self.eta * u - self.eta + 1, self.alpha))
        self.last_value = ret
        return ret

    def next_value(self):
        """
        Return the next value, skewed by the Zipfian distribution.
        """
        return self.next_long(self.items)

    def mean(self):
        """
        Calculate the mean of the distribution.
        """
        raise NotImplementedError("mean() method not implemented yet")

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Create a generator with 1000 items
    generator = ZipfianGenerator(max_val=16777216, zipfian_constant=0.999)
    
    # Generate samples for histogram testing
    # Generate 10000 samples to test the distribution
    samples = []
    num_samples = 10000
    
    print(f"Generating {num_samples} samples from Zipfian distribution...")
    for i in range(num_samples):
        samples.append(generator.next_value())
    
    # Create histogram
    plt.figure(figsize=(12, 8))
    
    # Plot histogram
    plt.subplot(2, 1, 1)
    plt.hist(samples, bins=50, alpha=0.7, edgecolor='black')
    plt.title(f'Zipfian Distribution Histogram (θ={generator.zipfian_constant})')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Plot log-log scale to show Zipfian property
    plt.subplot(2, 1, 2)
    counts, bins = np.histogram(samples, bins=50)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Remove zero counts for log plot
    non_zero_mask = counts > 0
    plt.loglog(bin_centers[non_zero_mask], counts[non_zero_mask], 'bo-', alpha=0.7)
    plt.title('Log-Log Plot (Zipfian Property Check)')
    plt.xlabel('Rank (log scale)')
    plt.ylabel('Frequency (log scale)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('zipfian_distribution.png')
    
    # Print some statistics
    print(f"\nStatistics:")
    print(f"Total samples: {len(samples)}")
    print(f"Min value: {min(samples)}")
    print(f"Max value: {max(samples)}")
    print(f"Most frequent value: {max(set(samples), key=samples.count)}")
    print(f"Unique values: {len(set(samples))}")
    
    # Show top 10 most frequent values
    counter = Counter(samples)
    print(f"\nTop 10 most frequent values:")
    for value, count in counter.most_common(10):
        print(f"Value {value}: {count} times ({count/len(samples)*100:.2f}%)")
    
    
