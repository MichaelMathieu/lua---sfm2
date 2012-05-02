#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTemplateType.hpp"
#else

//#define ARRAY_CHECKS

template<> class TH<real> {
public:
  typedef THTensorC CTensor;
  typedef THStorageC CStorage;
  typedef real THReal;
};

template<> class THStorage<real> {
public:
  typedef TH<real>::CTensor CTensor;
  typedef TH<real>::CStorage CStorage;
  typedef TH<real>::THReal THReal;
private:
  CStorage* cstorage;
public:
  inline THStorage(CStorage* cstorage)
    :cstorage(cstorage) {};
};

template<> class THTensor<real> {
public:
  typedef TH<real>::CTensor CTensor;
  typedef TH<real>::CStorage CStorage;
  typedef TH<real>::THReal THReal;
private:
  CTensor* ctensor;
  mutable bool hasToBeFreed;
public: // constructor etc.
  inline THTensor(CTensor* ctensor, bool hasToBeFreed = false)
    :ctensor(ctensor), hasToBeFreed(hasToBeFreed) {};
  inline THTensor(const THTensor & src)
    :ctensor(src.ctensor), hasToBeFreed(src.hasToBeFreed) {
    if (hasToBeFreed)
      retain();
  };
  inline ~THTensor() {
    if (hasToBeFreed) this->free();
  };
  inline THTensor & operator=(const THTensor & src) {
    if (&src != this) {
      if (hasToBeFreed)
	this->free();
      ctensor = src.ctensor;
      hasToBeFreed = src.hasToBeFreed;
      if (hasToBeFreed)
	retain();
    }
    return *this;
  };
public: // methods
  inline void retain() {
    THTensor_(retain)(ctensor);
  };
  inline void free() {
    THTensor_(free)(ctensor);
    hasToBeFreed = false;
  };
  inline THStorage<real> storage() {
    return THStorage<real>(THTensor_(storage)(ctensor));
  };
  inline THReal* data() {
    return THTensor_(data)(ctensor);
  };
  inline const THReal* data() const {
    return THTensor_(data)(ctensor);
  };
  inline long nDimension() const {
    return THTensor_(nDimension)(ctensor);
  };
  inline const long* stride() const {
    return ctensor->stride;
  };
  inline long stride(int i) const {
#ifdef ARRAY_CHECKS
    assert((0 <= i) && (i < nDimension()));
#endif
    return ctensor->stride[i];
  };
  inline const long* size() const {
    return ctensor->size;
  };
  inline long size(int i) const {
#ifdef ARRAY_CHECKS
    assert((0 <= i) && (i < nDimension()));
#endif
    return ctensor->size[i];
  };
  inline THTensor newContiguous() const {
    hasToBeFreed = true;
    return THTensor_(newContiguous)(ctensor);
  };
};

#endif
