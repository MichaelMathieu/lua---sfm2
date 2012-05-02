#include "genericpp/LM.hpp"

//HZ p605, Algo 6.1
// Finds P such that
// X = f(P) where P = (a_0 a_1 ... a_n b_0 b_1 ... b_n)
// and eps = X-X_hat minimizing eps.t() * sigma * eps (so sigma is the inverse of covariance)
// dA and dB are the partial derivatives : dA = df/da_i and dB = df/db_i
void PartitionedLM(const matf & X, const matf & initialGuessA,
		   const matf & initialGuessB,
		   matf(*f)(const matf &, const matf &),
		   matf(*dA)(const matf &), matf(*dB)(const matf &),
		   const matf & sigma, matf & a_out, matf & b_out) {
  float targetErr = 0.0000000000001f;
  float lambda = 0.001f;
  matf a = initialGuessA, b = initialGuessB;
  matf eps = X - (*f)(a, b);
  float oldErr = ((matf)(eps.t() * sigma * eps))(0,0), newErr;
  matf A, B, U, V, W, Y, epsA, epsB, deltaa, deltab, newa, newb;
  while (oldErr > targetErr) {
    cout << oldErr << endl;
    A = (*dA)(a);
    B = (*dB)(b);
    U = A.t() * sigma * A;
    V = B.t() * sigma * B;
    W = A.t() * sigma * B;
    epsA = A.t() * sigma * eps;
    epsB = B.t() * sigma * eps;
    do {
      for (int i = 0; i < U.size().height; ++i)
	U(i,i) *= 1.0f + lambda;
      for (int i = 0; i < V.size().height; ++i)
	V(i,i) *= 1.0f + lambda;
      V = V.inv(DECOMP_LU);
      Y = W * V;
      cout << lambda << endl;
      deltaa = (U - Y * W.t()).inv(DECOMP_LU) * (epsA - Y * epsB);
      deltab = V * (epsB - W.t() * deltaa);
      newa = a + deltaa;
      newb = b + deltab;
      eps = X - (*f)(newa, newb);
      newErr = ((matf)(eps.t() * sigma * eps))(0,0);
      lambda *= 10.0f;
    } while (newErr >= oldErr);
    lambda *= 0.01f;
    a = newa.clone();
    b = newb.clone();
    oldErr = newErr;
  } 
  a_out = a;
  b_out = b;
}

