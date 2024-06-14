#ifndef ABLATELIBRARY_GAUSSIANCONVOLUTION_HPP
#define ABLATELIBRARY_GAUSSIANCONVOLUTION_HPP

#include "domain/RBF/ga.hpp"
#include "domain/RBF/hybrid.hpp"
#include "domain/RBF/imq.hpp"
#include "domain/RBF/mq.hpp"
#include "domain/RBF/intMQ.hpp"
#include "domain/RBF/phs.hpp"
#include "domain/RBF/rbf.hpp"
#include "domain/subDomain.hpp"
#include "utilities/petscUtilities.hpp"



#include "domain/range.hpp"
#include "domain/reverseRange.hpp"



namespace ablate::levelSet {

  class GaussianConvolution {

    private:

      void BuildList(const PetscInt p);

      // Total number of cells used in the stencil.
      PetscInt nStencil = -1;

      // Number of points in the 1D quadrature
      PetscInt nQuad = -1;

      //   The quadrature is actually sqrt(2) times the quadrature points. This is as we are integrating
      //      against the normal distribution, not exp(-x^2)
      PetscReal *quad = nullptr;

      // The weights are the true weights divided by sqrt(pi)
      PetscReal *weights = nullptr;

      // The standard deviation distance
      PetscReal sigma = 1.0;

      PetscInt maxPoint = -1; // The number of vertices, edges, etc


      // List of cells necessary to do the integration.
      PetscInt **gaussCellList = nullptr;

      DM geomDM = nullptr;

      // Iteration range to use.
      PetscInt range[3] = {-1, -1, -1};

      // Number of neighboring cells to search when creating a list
      PetscInt nSearch = -1;



    public:

      void Evaluate(DM dm, std::shared_ptr<ablate::domain::rbf::RBF> rbf, const PetscInt fid, Vec fVec, PetscInt offset, const PetscInt p, const PetscInt nc, const PetscInt dx[], const PetscInt dy[], const PetscInt dz[], PetscReal vals[]);


      void Evaluate(DM dm, std::shared_ptr<ablate::domain::rbf::RBF> rbf, const PetscInt fid, const PetscScalar *array, PetscInt offset, const PetscInt p, const PetscInt nc, const PetscInt dx[], const PetscInt dy[], const PetscInt dz[], PetscReal vals[]);

      GaussianConvolution(DM geomDM, const PetscInt nQuad, const PetscInt sigmaFactor);


      ~GaussianConvolution();

  };



}  // namespace ablate::levelSet
#endif  // ABLATELIBRARY_GAUSSIANCONVOLUTION_HPP
