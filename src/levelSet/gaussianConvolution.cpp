#include "gaussianConvolution.hpp"
#include <petsc.h>
#include "utilities/petscSupport.hpp"
#include "utilities/mathUtilities.hpp"
#include "domain/fieldAccessor.hpp"
#include "levelSetUtilities.hpp"
#include "utilities/constants.hpp"



using namespace ablate::levelSet;

#define xexit(S, ...) {PetscFPrintf(MPI_COMM_WORLD, stderr, \
  "\x1b[1m(%s:%d, %s)\x1b[0m\n  \x1b[1m\x1b[90mexiting:\x1b[0m " S "\n", \
  __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__); exit(0);}

// geomDM - Sample DM with the geometry. Other DMs can store the data, but the geometric layout (including all ghost cells, etc) must match this exactly
// nQuad - The size of the quadrature. Valid values are [2, 10]. Recommendation is nQuad = 4;
// sigmaFactor - The standard deviation will be sigmaFactor*h. Recommendation is sigmaFactor = 1.0;

GaussianConvolution::GaussianConvolution(DM geomDM, const PetscInt nQuad, const PetscInt sigmaFactor) : nQuad(nQuad), geomDM(geomDM) {

  PetscReal weights1D[nQuad];

  PetscMalloc1(nQuad, &quad) >> ablate::utilities::PetscUtilities::checkError;

  switch (nQuad) {
    case 2:
      quad[0] = -1.00000000000000000000000000000;  weights1D[0]=0.500000000000000000000000000000;
      quad[1] =  1.00000000000000000000000000000;  weights1D[1]=0.500000000000000000000000000000;
      nSearch = 3;
      break;
    case 3:
      quad[0] = 0.0;                               weights1D[0]=0.666666666666666666666666666667;
      quad[1] = -1.73205080756887729352744634151;  weights1D[1]=0.166666666666666666666666666667;
      quad[2] =  1.73205080756887729352744634151;  weights1D[2]=0.166666666666666666666666666667;
      nSearch = 4;
      break;
    case 4:
      quad[0] = -2.33441421833897723931751226721;  weights1D[0]=0.0458758547680684918168929937745;
      quad[1] =  2.33441421833897723931751226721;  weights1D[1]=0.0458758547680684918168929937745;
      quad[2] = -0.741963784302725857648513596726; weights1D[2]=0.454124145231931508183107006225;
      quad[3] =  0.741963784302725857648513596726; weights1D[3]=0.454124145231931508183107006225;
      nSearch = 4;
      break;
    case 5:
      quad[0] = 0.0;                               weights1D[0]=0.533333333333333333333333333333;
      quad[1] = -2.85697001387280565416230426401;  weights1D[1]=0.0112574113277206889333702151856;
      quad[2] =  2.85697001387280565416230426401;  weights1D[2]=0.0112574113277206889333702151856;
      quad[3] = -1.35562617997426586583052129087;  weights1D[3]=0.222075922005612644399963118148;
      quad[4] =  1.35562617997426586583052129087;  weights1D[4]=0.222075922005612644399963118148;
      nSearch = 5;
      break;
    case 6:
      quad[0] = -0.616706590192594152193686099399; weights1D[0]=0.408828469556029226088537826250;
      quad[1] =  0.616706590192594152193686099399; weights1D[1]=0.408828469556029226088537826250;
      quad[2] = -1.88917587775371067550566789858;  weights1D[2]=0.0886157460419145274808558830057;
      quad[3] =  1.88917587775371067550566789858;  weights1D[3]=0.0886157460419145274808558830057;
      quad[4] = -3.32425743355211895236183546247;  weights1D[4]=0.00255578440205624643060629074383;
      quad[5] =  3.32425743355211895236183546247;  weights1D[5]=0.00255578440205624643060629074383;
      nSearch = 5;
      break;
    case 7:
      quad[0] =  0.0;                              weights1D[0]=0.457142857142857142857142857143;
      quad[1] = -1.15440539473996812723959775884;  weights1D[1]=0.240123178605012713740161995179;
      quad[2] =  1.15440539473996812723959775884;  weights1D[2]=0.240123178605012713740161995179;
      quad[3] = -2.36675941073454128861885646856;  weights1D[3]=0.0307571239675864970396450057164;
      quad[4] =  2.36675941073454128861885646856;  weights1D[4]=0.0307571239675864970396450057164;
      quad[5] = -3.75043971772574225630392202571;  weights1D[5]=0.000548268855972217791621570532802;
      quad[6] =  3.75043971772574225630392202571;  weights1D[6]=0.000548268855972217791621570532802;
      nSearch = 6;
      break;
    case 8:
      quad[0] = -0.539079811351375108072461918694; weights1D[0]=0.373012257679077349925549534301;
      quad[1] =  0.539079811351375108072461918694; weights1D[1]=0.373012257679077349925549534301;
      quad[2] = -1.63651904243510799922544657297;  weights1D[2]=0.117239907661759015117137525962;
      quad[3] =  1.63651904243510799922544657297;  weights1D[3]=0.117239907661759015117137525962;
      quad[4] = -2.80248586128754169911301080618;  weights1D[4]=0.00963522012078826718691913771988;
      quad[5] =  2.80248586128754169911301080618;  weights1D[5]=0.00963522012078826718691913771988;
      quad[6] = -4.14454718612589433206019783917;  weights1D[6]=0.000112614538375367770393802016870;
      quad[7] =  4.14454718612589433206019783917;  weights1D[7]=0.000112614538375367770393802016870;
      nSearch = 6;
      break;
    case 9:
      quad[0] =  0.0;                              weights1D[0]=0.406349206349206349206349206349;
      quad[1] = -1.02325566378913252482814822581;  weights1D[1]=0.244097502894939436141022017700;
      quad[2] =  1.02325566378913252482814822581;  weights1D[2]=0.244097502894939436141022017700;
      quad[3] = -2.07684797867783010652215614374;  weights1D[3]=0.0499164067652178740433414693826;
      quad[4] =  2.07684797867783010652215614374;  weights1D[4]=0.0499164067652178740433414693826;
      quad[5] = -3.20542900285646994336567590292;  weights1D[5]=0.00278914132123176862881344575164;
      quad[6] =  3.20542900285646994336567590292;  weights1D[6]=0.00278914132123176862881344575164;
      quad[7] = -4.51274586339978266756667884317;  weights1D[7]=0.0000223458440077465836484639907118;
      quad[8] =  4.51274586339978266756667884317;  weights1D[8]=0.0000223458440077465836484639907118;
      nSearch = 7;
      break;
    case 10:
      quad[0] = -0.484935707515497653046233483105; weights1D[0]=0.344642334932019042875028116518;
      quad[1] =  0.484935707515497653046233483105; weights1D[1]=0.344642334932019042875028116518;
      quad[2] = -1.46598909439115818325066466416;  weights1D[2]=0.135483702980267735563431657727;
      quad[3] =  1.46598909439115818325066466416;  weights1D[3]=0.135483702980267735563431657727;
      quad[4] = -2.48432584163895458087625118368;  weights1D[4]=0.0191115805007702856047383687629;
      quad[5] =  2.48432584163895458087625118368;  weights1D[5]=0.0191115805007702856047383687629;
      quad[6] = -3.58182348355192692277623675546;  weights1D[6]=0.000758070934312217670069636036508;
      quad[7] =  3.58182348355192692277623675546;  weights1D[7]=0.000758070934312217670069636036508;
      quad[8] = -4.85946282833231215015516494660;  weights1D[8]=4.31065263071828673222095472620e-6;
      quad[9] =  4.85946282833231215015516494660;  weights1D[9]=4.31065263071828673222095472620e-6;
      nSearch = 7;
    break;
    default:
      throw std::runtime_error("Only Gaussian-Hermite quadratures up to 10th-order are considered");
  }

  // The maximum number of elements (vertices, edges, etc).
  PetscInt depth;
  DMPlexGetDepth(geomDM, &depth) >> ablate::utilities::PetscUtilities::checkError;

  maxPoint = -1;
  for (PetscInt d = 0; d < depth; ++d) {
    PetscInt depthMax;
    DMPlexGetDepthStratum(geomDM, d, NULL, &depthMax) >> ablate::utilities::PetscUtilities::checkError;
    maxPoint = PetscMax(depthMax, maxPoint);
  }

  // Wait to allocate the actually memory for later
  PetscMalloc1(maxPoint, &gaussCellList);
  for (PetscInt p = 0; p < maxPoint; ++p) {
    gaussCellList[p] = nullptr;
  }

  // The spatial standard deviation to use.
  PetscReal h;
  DMPlexGetMinRadius(geomDM, &h) >> ablate::utilities::PetscUtilities::checkError;
  h *= 2.0;
  sigma = sigmaFactor*h;

  // The total number of cells in a stencil
  PetscInt dim;
  DMGetDimension(geomDM, &dim) >> ablate::utilities::PetscUtilities::checkError;
  nStencil = PetscPowInt(nQuad, dim); // The number of cells in the integration stencil

  // Ranges to iterate over
  range[0] = nQuad;
  range[1] = (dim > 1) ? nQuad : 1;
  range[2] = (dim > 2) ? nQuad : 1;

  // Now compute the stencil weight array. This is the same for all points
  PetscMalloc1(nStencil, &weights) >> ablate::utilities::PetscUtilities::checkError;
  for (PetscInt k = 0; k < range[2]; ++k) {
    for (PetscInt j = 0; j < range[1]; ++j) {
      for (PetscInt i = 0; i < range[0]; ++i) {
        weights[nQuad*(k*nQuad + j) + i] =  (dim > 2) ? weights1D[k] : 1.0;
        weights[nQuad*(k*nQuad + j) + i] *= (dim > 1) ? weights1D[j] : 1.0;
        weights[nQuad*(k*nQuad + j) + i] *=             weights1D[i];
      }
    }
  }

}

GaussianConvolution::~GaussianConvolution() {
  PetscFree(quad) >> ablate::utilities::PetscUtilities::checkError;

  for (PetscInt p = 0; p < maxPoint; ++p) {
    PetscFree(gaussCellList[p]) >> ablate::utilities::PetscUtilities::checkError;
  }
  PetscFree(gaussCellList) >> ablate::utilities::PetscUtilities::checkError;

}


PetscReal derivativeFactor(const PetscReal *x, const PetscReal s,  const PetscInt dx, const PetscInt dy, const PetscInt dz) {

  const PetscReal s2 = PetscSqr(s);

  const PetscInt derHash = 100*dx + 10*dy + dz;

  if (derHash > 0 && PetscAbsReal(s)<PETSC_SMALL) return (0.0);

  switch (derHash) {
    case   0: // Value
      return (1.0);
    case 100: // x
      return (x[0]/s2);
    case  10: // y
      return (x[1]/s2);
    case   1: // z
      return (x[2]/s2);
    case 200: // xx
      return ((x[0]*x[0] - s2)/PetscSqr(s2));
    case  20: // yy
      return ((x[1]*x[1] - s2)/PetscSqr(s2));
    case   2: // zz
      return ((x[2]*x[2] - s2)/PetscSqr(s2));
    case 110: // xy
      return (x[0]*x[1]/PetscSqr(s2));
    case 101: // xz
      return (x[0]*x[2]/PetscSqr(s2));
    case  11: // yz
      return (x[1]*x[2]/PetscSqr(s2));
    default:
      throw std::runtime_error("Unknown derivative request");
  }

}


static PetscInt FindCell(DM dm, const PetscInt dim, const PetscReal x0[], const PetscInt nCells, const PetscInt cells[]) {
  // Return the cell with the cell-center that is the closest to a given point
  for (PetscInt c = 0; c < nCells; ++c) {
    if (cells[c] > -1) {
      PetscBool inCell = PETSC_FALSE;
      DMPlexInCell(dm, cells[c], x0, &inCell) >> ablate::utilities::PetscUtilities::checkError;
      if (inCell) return cells[c];
    }
  }
  return -1;
}

// Build the list of cells needed for point p.
void GaussianConvolution::BuildList(const PetscInt p) {

  PetscInt dim;
  DMGetDimension(geomDM, &dim) >> ablate::utilities::PetscUtilities::checkError;

  PetscReal x0[3] = {0.0, 0.0, 0.0};
  DMPlexComputeCellGeometryFVM(geomDM, p, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

  PetscInt nCells, *cellList;
  DMPlexGetNeighbors(geomDM, p, nSearch, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cellList) >> ablate::utilities::PetscUtilities::checkError;

  PetscMalloc1(nStencil, &gaussCellList[p]);

  for (PetscInt i = 0; i < range[0]; ++i) {
    for (PetscInt j = 0; j < range[1]; ++j) {
      for (PetscInt k = 0; k < range[2]; ++k) {

        PetscReal x[3] = {x0[0] + sigma*quad[i], x0[1] + sigma*quad[j], x0[2] + sigma*quad[k]};

        gaussCellList[p][nQuad*(k*nQuad + j) + i] = FindCell(geomDM, dim, x, nCells, cellList);
      }
    }
  }

  DMPlexRestoreNeighbors(geomDM, p, 3, -1.0, -1, PETSC_FALSE, PETSC_FALSE, &nCells, &cellList) >> ablate::utilities::PetscUtilities::checkError;



}


void GaussianConvolution::Evaluate(DM dm, std::shared_ptr<ablate::domain::rbf::RBF> rbf, const PetscInt fid, Vec fVec, PetscInt offset, const PetscInt p, const PetscInt nc, const PetscInt dx[], const PetscInt dy[], const PetscInt dz[], PetscReal vals[]) {
  const PetscScalar *array;
  VecGetArrayRead(fVec, &array) >> ablate::utilities::PetscUtilities::checkError;
  Evaluate(dm, rbf, fid, array, offset, p, nc, dx, dy, dz, vals);
  VecRestoreArrayRead(fVec, &array) >> ablate::utilities::PetscUtilities::checkError;
}




// dm - The mesh
// rbf - The RBF used to interpolate. Must be compatible with the data in array (e.g. if array is vertex-based then the RBF must respect that)
// fid -
void GaussianConvolution::Evaluate(DM dm, std::shared_ptr<ablate::domain::rbf::RBF> rbf, const PetscInt fid, const PetscScalar *array, PetscInt offset, const PetscInt p, const PetscInt nc, const PetscInt dx[], const PetscInt dy[], const PetscInt dz[], PetscReal vals[]) {

  if (!gaussCellList[p]) BuildList(p);  // Build the convolution list

  PetscReal x0[3] = {0.0, 0.0, 0.0};
  DMPlexComputeCellGeometryFVM(geomDM, p, NULL, x0, NULL) >> ablate::utilities::PetscUtilities::checkError;

  PetscArrayzero(vals, nc) >> ablate::utilities::PetscUtilities::checkError;

  // Near the edges of the domain there may not be enough cells to do a true convolution.
  //  One possibility is to shift the integration window so that it is completely iside the domain. Might do this in the future.
  //  For now just normalize the values based on the total weight
  PetscReal totalWT = 0.0;

  for (PetscInt k = 0; k < range[2]; ++k) {
    for (PetscInt j = 0; j < range[1]; ++j) {
      for (PetscInt i = 0; i < range[0]; ++i) {

        const PetscInt pt = nQuad*(k*nQuad + j) + i;
        const PetscInt interpCell = gaussCellList[p][pt];

        if (interpCell > -1) {
          const PetscReal dist[3] = {sigma*quad[i], sigma*quad[j], sigma*quad[k]};
          PetscReal x[3] = {x0[0] + dist[0], x0[1] + dist[1], x0[2] + dist[2]};

          const PetscReal fVal = rbf->Interpolate(dm, fid, array, offset, interpCell, x);
          const PetscReal wt = weights[pt];

          totalWT += wt;

          for (PetscInt c = 0; c < nc; ++c) {
            vals[c] += wt*derivativeFactor(dist, sigma, dx[c], dy[c], dz[c])*fVal;
          }
        }
      }
    }
  }

  for (PetscInt c = 0; c < nc; ++c) vals[c] /= totalWT;

}
