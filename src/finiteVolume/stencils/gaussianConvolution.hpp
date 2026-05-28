#ifndef ABLATELIBRARY_GAUSSIANCONVOLUTION_HPP
#define ABLATELIBRARY_GAUSSIANCONVOLUTION_HPP

#include "utilities/petscSupport.hpp"
#include "utilities/petscUtilities.hpp"
#include <petsc/private/hashmapi.h>


// Cell-based gaussian convolution
namespace ablate::finiteVolume::stencil {

  class GaussianConvolution {

    private:

      PetscInt rangeStart;
      PetscInt rangeEnd;

      PetscInt dim;

      void BuildList(const PetscInt p);



      // The standard deviation distance squared
      PetscReal sigmaSqr = 1.0;

      // Factor used to multipy the exponential
      PetscReal fac = 0.0;

      // List of cells necessary to do the integration.
      PetscInt **cellList = nullptr;
      PetscInt *nCellList = nullptr;
      PetscReal **cellDist = nullptr;

      // (Base) Weights of each cell
      PetscReal **cellWeights = nullptr;

      // Derivative weights for each cell.
      // derWeights[k][p] gives the cell weights for the kth-derivative at point p
      std::vector<std::vector<std::vector<PetscReal>>> derWeights;


      const PetscInt keyFactors[3] = {100, 10, 1};
      PetscInt derivativeKey(const PetscInt dim, const PetscInt dx[]) {
        if (!dx) return 0; // If dx==NULL then it's the value, not a derivative
        PetscInt key = 0;
        for (PetscInt i = 0; i < dim; ++i) key += keyFactors[i] * dx[i];
        return key;
      }

      // Hash of the derivative: The location of derivative (dx, dy, dz) will be the kth-location in derWeights
      PetscHMapI derHash = nullptr;



      DM geomDM = nullptr;

       // Used for periodicity
      PetscReal maxDist[3] = {PETSC_MAX_REAL, PETSC_MAX_REAL, PETSC_MAX_REAL};
      PetscReal sideLen[3] = {0, 0, 0};

      const PetscInt dataDepth = -1;
      PetscInt searchDepth = -1;


    public:
      void Evaluate(const PetscInt p, const PetscInt dx[], DM dataDM, const PetscInt fid, const PetscScalar *array, PetscInt offset, const PetscInt nDof, PetscReal *vals);
      void Evaluate(const PetscInt p, const PetscInt dx[], DM dataDM, const PetscInt fid, Vec fVec, const PetscInt offset, const PetscInt nDof, PetscReal *vals);
      void Gradient(const PetscInt p, DM dataDM, const PetscInt fid, const PetscScalar *array, const PetscInt offset, const PetscInt nDof, PetscReal *vals);


      PetscInt GetCellList(const PetscInt p, const PetscInt **cellListOut);

      void FormAllLists();

      GaussianConvolution(DM geomDM, const PetscReal sigmaFactor, const PetscInt evalDepth, const PetscInt dataDepth);



      ~GaussianConvolution();

  };



}  // namespace ablate::levelSet
#endif  // ABLATELIBRARY_GAUSSIANCONVOLUTION_HPP
