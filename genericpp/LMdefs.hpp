template<typename Function>
void LM(const matf & X, const matf & initialGuess, const Function & func,
	const matf & sigma, matf & a, int max_iter, float target_err) {
  float lambda = 0.001f;
  a = initialGuess;
  matf eps = X - func.f(a);
  float oldErr = ((matf)(eps.t() * sigma * eps))(0,0), newErr;
  matf A, U, epsa, newa;
  int iter = max_iter+1;
  while ((oldErr > target_err) && (--iter)) {
    //cout << "Error " << oldErr << endl;
    A = func.dA(a);
    U = A.t() * sigma * A;
    epsa = A.t() * sigma * eps;
    do {
      //cout << "Lambda " << lambda << endl;
      if ((lambda > 1e6f) || (lambda < 1e-25f))
	return;
      for (int i = 0; i < U.size().height; ++i)
	U(i,i) *= 1.0f + lambda;
      newa = a + U.inv(DECOMP_CHOLESKY) * epsa;
      eps = X - func.f(newa);
      newErr = ((matf)(eps.t() * sigma * eps))(0,0);
      lambda *= 10.0f;
    } while (newErr >= oldErr);
    lambda *= 0.01f;
    a = newa.clone(); //TODO maybe not optimal
    oldErr = newErr;
  }
}

template<typename Function>
void LM(const matf & X, const matf & initialGuess, const Function & func, matf & a,
	int max_iter, float target_err) {
  float lambda = 0.001f;
  a = initialGuess;
  matf eps = X - func.f(a);
  float oldErr = ((matf)(eps.t() *  eps))(0,0), newErr;
  matf A, U, epsa, newa;
  int iter = max_iter+1;
  while ((oldErr > target_err) && (--iter)) {
    //cout << "Error " << oldErr << endl;
    A = func.dA(a);
    U = A.t() * A;
    epsa = A.t() * eps;
    do {
      //cout << "Lambda " << lambda << endl;
      if ((lambda > 1e6f) || (lambda < 1e-25f))
	return;
      for (int i = 0; i < U.size().height; ++i)
	U(i,i) *= 1.0f + lambda;
      newa = a + U.inv(DECOMP_CHOLESKY) * epsa;
      eps = X - func.f(newa);
      newErr = ((matf)(eps.t() * eps))(0,0);
      lambda *= 10.0f;
    } while (newErr >= oldErr);
    lambda *= 0.01f;
    a = newa.clone(); //TODO maybe not optimal
    oldErr = newErr;
  }
}

//HZ p608, Algo 6.3
// Sparse Levenberg-Marquardt algorithm (non linear error minimization)
//TODO put b in a single matf (multi line)
template<typename Function>
void SparseLM(const matf & X, const matf & initialGuess_a,
	      const matf & initialGuess_b,
	      const Function & func,
	      const vector<matf> & sigma, matf & a, matf & b) {
  float targetErr = 1e-5f;
  int n = X.size().height, i, j;
  float lambda = 0.001f;
  a = initialGuess_a;
  b = initialGuess_b;
  int nA = a.size().height;
  matf eps(X.size());
  float oldErr = 0.0f, newErr;
  for (i = 0; i < n; ++i) {
    eps.row(i) = X.row(i) - func.f(a, b.row(i).t(), i).t();
    oldErr += ((matf)(eps.row(i) * sigma[i] * eps.row(i).t()))(0,0);
  }
  matf U(nA, nA), epsA(nA, 1), deltaa, tmpA(nA, nA), tmpA2(nA, 1), Ai, Bi, tmp;
  vector<matf> Y, V, W, epsB;
  matf newa[2];
  matf newb[2];
  newb[0] = matf(b.size());
  newb[1] = matf(b.size());
  int inew = 0; //avoid reallocations at each loop
  for (i = 0; i < n; ++i) {
    Y.push_back(matf());
    V.push_back(matf());
    W.push_back(matf());
    epsB.push_back(matf());
  }

  while (oldErr > targetErr) {
    //cout << "Error: " << oldErr << endl;
    U.setTo(0.0f);
    epsA.setTo(0.0f);
    for (i = 0; i < n; ++i) {
      Ai = func.dA(a, b.row(i).t(), i);
      Bi = func.dB(a, b.row(i).t(), i);
      U += Ai.t() * sigma[i] * Ai;
      V[i] = Bi.t() * sigma[i] * Bi;
      W[i] = Ai.t() * sigma[i] * Bi;
      epsA += Ai.t() * sigma[i] * eps.row(i).t();
      epsB[i] = Bi.t() * sigma[i] * eps.row(i).t();
    }
    do {
      if ((lambda > 1e25) || (lambda < 1e-25))
	return;
      //cout << "lambda: " << lambda << endl;
      for (i = 0; i < U.size().height; ++i)
	U(i,i) *= 1.0f + lambda;
      for (i = 0; i < n; ++i) {
	for (j = 0; j < V[i].size().height; ++j)
	  V[i](j,j) *= 1.0f + lambda;
	V[i] = V[i].inv(DECOMP_CHOLESKY);
	Y[i] = W[i] * V[i];
      }
      tmpA.setTo(0.0f);
      tmpA2.setTo(0.0f);
      for (i = 0; i < n; ++i) {
	tmpA += Y[i] * W[i].t();
	tmpA2 += Y[i] * epsB[i];
      }
      deltaa = (U - tmpA).inv(DECOMP_LU) * (epsA - tmpA2);
      for (i = 0; i < n; ++i) {
	copyRow(b.row(i) + (V[i] * (epsB[i] - W[i].t() * deltaa)).t(), newb[inew], 0, i);
      }
      newa[inew] = a + deltaa;
      newErr = 0.0f;
      for (i = 0; i < n; ++i) {
	copyRow(X.row(i) - func.f(newa[inew], newb[inew].row(i).t(), i).t(), eps, 0, i);
	newErr += ((matf)(eps.row(i) * sigma[i] * eps.row(i).t()))(0,0);
      }
      lambda *= 10.0f;
    } while (newErr >= oldErr);
    if (isnan(newErr))
      return;
    lambda *= 0.01f;
    tmp = a;
    a = newa[inew];
    newa[1-inew] = tmp;
    tmp = b;
    b = newb[inew];
    newb[1-inew] = tmp;
    inew = 1-inew;
    oldErr = newErr;
  }
}

template<typename Function>
void SparseLM(const matf & X, const matf & initialGuess_a,
	      const matf & initialGuess_b,
	      const Function & func, matf & a, matf & b) {
  float targetErr = 1e-5f;
  int n = X.size().height, i, j;
  float lambda = 0.001f;
  a = initialGuess_a;
  b = initialGuess_b;
  int nA = a.size().height;
  matf eps(X.size());
  float oldErr = 0.0f, newErr;
  for (i = 0; i < n; ++i) {
    eps.row(i) = X.row(i) - func.f(a, b.row(i).t(), i).t();
    oldErr += ((matf)(eps.row(i) * eps.row(i).t()))(0,0);
  }
  matf U(nA, nA), epsA(nA, 1), deltaa, tmpA(nA, nA), tmpA2(nA, 1), Ai, Bi, tmp;
  vector<matf> Y, V, W, epsB;
  matf newa[2];
  matf newb[2];
  newb[0] = matf(b.size());
  newb[1] = matf(b.size());
  int inew = 0; //avoid reallocations at each loop
  for (i = 0; i < n; ++i) {
    Y.push_back(matf());
    V.push_back(matf());
    W.push_back(matf());
    epsB.push_back(matf());
  }

  while (oldErr > targetErr) {
    //cout << "Error: " << oldErr << endl;
    U.setTo(0.0f);
    epsA.setTo(0.0f);
    for (i = 0; i < n; ++i) {
      Ai = func.dA(a, b.row(i).t(), i);
      Bi = func.dB(a, b.row(i).t(), i);
      U += Ai.t() * Ai;
      V[i] = Bi.t() * Bi;
      W[i] = Ai.t() * Bi;
      epsA += Ai.t() * eps.row(i).t();
      epsB[i] = Bi.t() * eps.row(i).t();
    }
    do {
      if ((lambda > 1e25) || (lambda < 1e-25))
	return;
      //cout << "lambda: " << lambda << endl;
      for (i = 0; i < U.size().height; ++i)
	U(i,i) *= 1.0f + lambda;
      for (i = 0; i < n; ++i) {
	for (j = 0; j < V[i].size().height; ++j)
	  V[i](j,j) *= 1.0f + lambda;
	V[i] = V[i].inv(DECOMP_CHOLESKY);
	Y[i] = W[i] * V[i];
      }
      tmpA.setTo(0.0f);
      tmpA2.setTo(0.0f);
      for (i = 0; i < n; ++i) {
	tmpA += Y[i] * W[i].t();
	tmpA2 += Y[i] * epsB[i];
      }
      deltaa = (U - tmpA).inv(DECOMP_LU) * (epsA - tmpA2);
      for (i = 0; i < n; ++i) {
	copyRow(b.row(i) + (V[i] * (epsB[i] - W[i].t() * deltaa)).t(), newb[inew], 0, i);
      }
      newa[inew] = a + deltaa;
      newErr = 0.0f;
      for (i = 0; i < n; ++i) {
	copyRow(X.row(i) - func.f(newa[inew], newb[inew].row(i).t(), i).t(), eps, 0, i);
	newErr += ((matf)(eps.row(i) * eps.row(i).t()))(0,0);
      }
      lambda *= 10.0f;
    } while (newErr >= oldErr);
    if (isnan(newErr))
      return;
    lambda *= 0.01f;
    tmp = a;
    a = newa[inew];
    newa[1-inew] = tmp;
    tmp = b;
    b = newb[inew];
    newb[1-inew] = tmp;
    inew = 1-inew;
    oldErr = newErr;
  }
}
