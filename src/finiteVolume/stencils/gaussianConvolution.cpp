#include "gaussianConvolution.hpp"
#include <petsc.h>
#include "utilities/petscSupport.hpp"
#include "utilities/mathUtilities.hpp"
#include "domain/fieldAccessor.hpp"
#include "utilities/constants.hpp"

using namespace ablate::finiteVolume::stencil;

#define xexit(S, ...) {PetscFPrintf(MPI_COMM_WORLD, stderr, \
  "\x1b[1m(%s:%d, %s)\x1b[0m\n  \x1b[1m\x1b[90mexiting:\x1b[0m " S "\n", \
  __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__); exit(0);}

// Gaussian convolution of cell-centered data
//
// geomDM - Sample DM with the geometry. Other DMs can store the data, but the geometric layout (including all ghost cells, etc) must match this exactly
// nLayers - The number of layers to use. Recommendation is nQuad = 4;
// sigmaFactor - The standard deviation will be sigmaFactor*h. Recommendation is sigmaFactor = 1.0;
// pointLoc - The depth or height of the points where the convolution will be calculated
// d_or_h - Indicates if pointLoc is the depth or height
GaussianConvolution::GaussianConvolution(DM geomDM, const PetscReal sigmaFactor, const PetscInt evalDepth, const PetscInt dataDepth) : geomDM(geomDM), dataDepth(dataDepth) {

  PetscInt dim;
  DMGetDimension(geomDM, &dim);

  if (dataDepth!=dim) {
    std::runtime_error("GaussianConvolution really need to be re-evaluated for non-cell based data. There is no `volume` in the traditional sense.");
  }

  DMPlexGetDepthStratum(geomDM, evalDepth, &rangeStart, &rangeEnd) >> utilities::PetscUtilities::checkError;

  PetscInt nRange = rangeEnd - rangeStart;

  PetscMalloc4(nRange, &nCellList, nRange, &cellList, nRange, &cellWeights, nRange, &cellDist) >> utilities::PetscUtilities::checkError;

  nCellList -= rangeStart; // So that it can be index by cell number
  cellList -= rangeStart;
  cellWeights -= rangeStart;
  cellDist -= rangeStart;
  for (PetscInt c = rangeStart; c < rangeEnd; ++c) {
    nCellList[c] = -1;
    cellList[c] = nullptr;
    cellWeights[c] = nullptr;
    cellDist[c] = nullptr;
  }

  // The spatial standard deviation to use.
  PetscReal h;
  DMPlexGetMinRadius(geomDM, &h) >> ablate::utilities::PetscUtilities::checkError;
  h *= 2.0;
  PetscReal sigma = sigmaFactor*h;

  fac = PetscSqrtReal(2.0*PETSC_PI)*sigma;
  fac = PetscPowRealInt(fac, -dim);
  sigmaSqr = PetscSqr(sigma);


  // Get the information about periodicity
  const PetscReal *maxCell, *L;
  DMGetPeriodicity(geomDM, &maxCell, NULL, &L) >> ablate::utilities::PetscUtilities::checkError;

  if (maxCell) { // If maxCell==NULL then there are no periodic sides
    for (PetscInt d = 0; d < dim; ++d) {
      if (maxCell[d] > 0.0) {
        // Assume that we will be using all cells within 3 standard deviations of a center point.
        // Make the maximum distance check twice that
        maxDist[d] = 6*sigmaFactor*maxCell[d];
        sideLen[d] = L[d];
      }
    }
  }

  switch (dataDepth) {
    case 0: // vertices
    case 1: // edges
      searchDepth = dim;
      break;
    case 2: // faces
    case 3: // cells
      searchDepth = 0;
      break;
    default:
      throw std::runtime_error("Unknown data depth");
  }

}

GaussianConvolution::~GaussianConvolution() {


  for (PetscInt c = rangeStart; c < rangeEnd; ++c) {
    PetscFree3(cellList[c], cellWeights[c], cellDist[c]) >> utilities::PetscUtilities::checkError;
  }
  nCellList += rangeStart;
  cellList += rangeStart;
  cellWeights += rangeStart;
  cellDist += rangeStart;
  PetscFree4(nCellList, cellList, cellWeights, cellDist) >> utilities::PetscUtilities::checkError;
}

PetscInt derivativeHash(const PetscInt dim, const PetscInt dx[]) {
  PetscInt derHash = 100*dx[0];
  if (dim>1) derHash += 10*dx[1];
  if (dim>2) derHash += dx[2];

  return derHash;
}

PetscReal derivativeFactor(const PetscReal *x, const PetscReal sigmaSqr, const PetscInt derHash) {

  if (derHash > 0 && sigmaSqr<PETSC_SMALL) return (0.0);

  switch (derHash) {
    case   0: // Value
      return (1.0);
    case 100: // x
      return (x[0]/sigmaSqr);
    case  10: // y
      return (x[1]/sigmaSqr);
    case   1: // z
      return (x[2]/sigmaSqr);
    case 200: // xx
      return ((x[0]*x[0] - sigmaSqr)/PetscSqr(sigmaSqr));
    case  20: // yy
      return ((x[1]*x[1] - sigmaSqr)/PetscSqr(sigmaSqr));
    case   2: // zz
      return ((x[2]*x[2] - sigmaSqr)/PetscSqr(sigmaSqr));
    case 110: // xy
      return (x[0]*x[1]/PetscSqr(sigmaSqr));
    case 101: // xz
      return (x[0]*x[2]/PetscSqr(sigmaSqr));
    case  11: // yz
      return (x[1]*x[2]/PetscSqr(sigmaSqr));
    default:
      printf("%ld\n", derHash);
      throw std::runtime_error("Unknown derivative request");
  }

}


// Build the list of cells needed for point p.
// Note that this will return boundary cells
void GaussianConvolution::BuildList(const PetscInt p) {

  PetscInt  dim;
  PetscReal x0[3];
  PetscInt  nLocalCellList, *localCellList;

  DMGetDimension(geomDM, &dim);

  DMPlexGetNeighborsNew(geomDM, p, 4*PetscSqrtReal(sigmaSqr), DMPLEX_NEIGHBOR_MAXDIST, searchDepth, dataDepth, &nLocalCellList, &localCellList) >> ablate::utilities::PetscUtilities::checkError;



  PetscMalloc3(nLocalCellList, &cellList[p], nLocalCellList, &cellWeights[p], dim*nLocalCellList, &cellDist[p]) >> ablate::utilities::PetscUtilities::checkError;

  DMPlexPointGeometricData(geomDM, p, NULL, x0, NULL) >> utilities::PetscUtilities::checkError;

  PetscInt nnz = 0; // The number of non-zero entries
  PetscReal totalWt = 0.0;
  for (PetscInt n = 0; n < nLocalCellList; ++n) {
    PetscInt neighborCell = localCellList[n];
    PetscReal vol, x[3];
    DMPlexPointGeometricData(geomDM, neighborCell, &vol, x, NULL) >> utilities::PetscUtilities::checkError;

    // Compute the distance, taking into account periodicity
    PetscReal r = 0.0, dist[3] = {0.0, 0.0, 0.0};
    for (PetscInt d = 0; d < dim; ++d) {
      dist[d] = x[d] - x0[d];
      if (dist[d] > maxDist[d]){
        dist[d] -= sideLen[d];
      }
      else if(dist[d] < -maxDist[d]){
        dist[d] += sideLen[d];
      }
      r += PetscSqr(dist[d]);
    }

    PetscReal wt = (vol)*fac*PetscExpReal(-0.5*r/sigmaSqr);

    if (wt > PETSC_SMALL) {
      cellList[p][nnz] = neighborCell;
      cellWeights[p][nnz] = wt;
      for (PetscInt d = 0; d < dim; ++d) cellDist[p][nnz*dim + d] = dist[d];
      ++nnz;
      totalWt += wt;
    }
  }
  nCellList[p] = nnz;

  for (PetscInt n = 0; n < nnz; ++n) cellWeights[p][n] /= totalWt;

  DMPlexRestoreNeighborsNew(geomDM, p, 4*PetscSqrtReal(sigmaSqr), DMPLEX_NEIGHBOR_MAXDIST, searchDepth, dataDepth, &nLocalCellList, &localCellList) >> ablate::utilities::PetscUtilities::checkError;


}


void GaussianConvolution::Evaluate(const PetscInt p, const PetscInt dx[], DM dataDM, const PetscInt fid, Vec fVec, PetscInt offset, const PetscInt nDof, PetscReal *vals) {
  const PetscScalar *array;
  VecGetArrayRead(fVec, &array) >> ablate::utilities::PetscUtilities::checkError;
  Evaluate(p, dx, dataDM, fid, array, offset, nDof, vals);
  VecRestoreArrayRead(fVec, &array) >> ablate::utilities::PetscUtilities::checkError;
}

void GaussianConvolution::FormAllLists() {
  for (PetscInt cell = GaussianConvolution::rangeStart; cell < GaussianConvolution::rangeEnd; ++cell){
    BuildList(cell);
  }
}
PetscInt GaussianConvolution::GetCellList(const PetscInt p, const PetscInt **cellListOut) {

  if (!cellList[p]) BuildList(p);  // Build the convolution list

  *(const PetscInt **)cellListOut = cellList[p];
  return nCellList[p];
}



// p - Center cell of interest
// dx - Derivatives in the [x, y, z]-directions
// dataDM - DM containing the data
// fid - field id
// array - Array of the data
// offset - Where the data of interest starts
// nDof - Number of degrees of freedom, i.e. number of components in the vector
// vals - Smoothed values
void GaussianConvolution::Evaluate(const PetscInt p, const PetscInt dx[], DM dataDM, const PetscInt fid, const PetscScalar *array, PetscInt offset, const PetscInt nDof, PetscReal *vals) {

  if (p < rangeStart || p >= rangeEnd) {
    throw std::runtime_error("Attempting to calculate a gaussian convolution at a point outside of the specified range.");
  }

  if (!cellList[p]) BuildList(p);  // Build the convolution list

  PetscInt dim;
  DMGetDimension(geomDM, &dim);
  PetscInt derHash = 0;
  if (dx) derHash = derivativeHash(dim, dx);

  PetscArrayzero(vals, nDof) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt i = 0; i < nCellList[p]; ++i) {
    PetscInt cell = cellList[p][i];

    PetscReal vol, x[3];
    DMPlexPointGeometricData(geomDM, cell, &vol, x, NULL) >> utilities::PetscUtilities::checkError;

    const PetscScalar *data;
    xDMPlexPointLocalRead(dataDM, cell, fid, array, &data);

    const PetscReal derFac = derivativeFactor(&cellDist[p][i*dim], sigmaSqr, derHash);

    for (PetscInt c = 0; c < nDof; ++c) {
      vals[c] += derFac*data[offset + c]*cellWeights[p][i];
    }
  }
}
