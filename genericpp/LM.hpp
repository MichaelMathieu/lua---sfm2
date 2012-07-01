#ifndef __LM_H__
#define __LM_H__

#include "genericpp/common.hpp"

template<typename Function>
void LM(const matf & X, const matf & initialGuess, const Function & func,
	const matf & sigma, matf & a_out, int max_iter = 10000, float target_err = 1e-5);

template<typename Function>
void LM(const matf & X, const matf & initialGuess, const Function & func, matf & a_out,
	int max_iter = 10000, float target_err = 1e-5);

//HZ p605, Algo 6.1
// Finds P such that
// X_hat = f(P) where P = (a_0 a_1 ... a_n b_0 b_1 ... b_m)
// and eps = X-X_hat minimizing eps.t() * sigma * eps (so sigma is the inverse of covariance)
// dA and dB are the partial derivatives : dA = df/da_i and dB = df/db_i
void PartitionedLM(const matf & X, const matf & initialGuessA,
		   const matf & initialGuessB,
		   matf(*f)(const matf &, const matf &),
		   matf(*dA)(const matf &), matf(*dB)(const matf &),
		   const matf & covariance, matf & a_out, matf & b_out);

//HZ p608, Algo 6.3
// Sparse Levenberg-Marquardt algorithm (non linear error minimization)
// Finds P such that
// X_hat_i = f(P) where P = (a_0 a_1 ... a_n b_i_0 b_i_1 ... b_i_mi)
// and eps_i = X_i-X_hat_i minimizing sum(eps_i.t() * sigma * eps_i)
// (sigma = inverse of covariance)
// dA and dB are the partial derivatives : dA(a, i) = dX_i/da_j and dB(b_i, i) = dX_i/db_j
// (a_0 .. a_n) and (b_0 b_1 ... b_n) corresponding to the coordinates of X_j
// ( dX_i/db_j = 0 for other j, but not dX_i/da_j )
// Each sigma must be symetric (can be changed)
template<typename Function>
void SparseLM(const matf & X, const matf & initialGuess_a,
	      const matf & initialGuess_b,
	      const Function & func,
	      const vector<matf> & sigma, matf & a_out, matf & b_out);
template<typename Function>
void SparseLM(const matf & X, const matf & initialGuess_a, const matf & initialGuess_b,
	      const Function & func, matf & a_out, matf & b_out);

#include "LMdefs.hpp"

#endif
