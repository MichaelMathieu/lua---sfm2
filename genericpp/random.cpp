#include "random.hpp"
#include<ext/algorithm>
using namespace std;

class GRSDummyIterator {
private:
  int n;
public:
  typedef forward_iterator_tag iterator_category;
  typedef int value_type;
  typedef int difference_type;
  typedef int reference;
  typedef int* pointer;
  inline GRSDummyIterator(int n) :n(n) {};
  inline GRSDummyIterator& operator++() {++n; return *this;};
  inline bool operator!=(const GRSDummyIterator & other) {return n != other.n;};
  inline reference operator*() const {return n;};
};

void GetRandomSample(vector<size_t> & sample, size_t a, size_t b) {
  __gnu_cxx::random_sample(GRSDummyIterator(a), GRSDummyIterator(b),
			   sample.begin(), sample.end());
}
