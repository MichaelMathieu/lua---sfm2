#ifndef __EPIPOLES_H__
#define __EPIPOLES_H__

#include "genericpp/common.hpp"

matf GetEpipoleFromLinesSVD(const matf & lines);
matf GetEpipoleFromLinesRansac(const matf & lines, float d = 1.0f, float p = 0.99f);

#endif
