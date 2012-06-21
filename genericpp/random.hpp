#ifndef __RANDOM_H_SFM2__
#define __RANDOM_H_SFM2__

#include<vector>

// returns random sample of size sample.size() between a (included) and b (not included)
void GetRandomSample(std::vector<std::size_t> & sample, std::size_t a, std::size_t b);

#endif
