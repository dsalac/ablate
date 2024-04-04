#ifndef ABLATELIBRARY_INTERFACERECONSTRUCTION_HPP
#define ABLATELIBRARY_INTERFACERECONSTRUCTION_HPP

#include "domain/RBF/ga.hpp"
#include "domain/RBF/hybrid.hpp"
#include "domain/RBF/imq.hpp"
#include "domain/RBF/mq.hpp"
#include "domain/RBF/phs.hpp"
#include "domain/RBF/rbf.hpp"
#include "domain/subDomain.hpp"
#include "utilities/petscUtilities.hpp"



#include "domain/range.hpp"
#include "domain/reverseRange.hpp"



namespace ablate::levelSet {

  class Reconstruction {

    private:

      enum VecLoc { LOCAL , GLOBAL };


      static inline void UpdateVec(DM dm, Vec lv, Vec gv, PetscScalar **array) {
        VecRestoreArray(lv, array) >> utilities::PetscUtilities::checkError;
        DMLocalToGlobal(dm, lv, INSERT_VALUES, gv) >> utilities::PetscUtilities::checkError;
        DMGlobalToLocal(dm, gv, INSERT_VALUES, lv) >> utilities::PetscUtilities::checkError;
        VecGetArray(lv, array);
      }

      void BuildInterpCellList();

      //   Hermite-Gauss quadrature points
      const PetscInt nQuad = 4; // Size of the 1D quadrature
      //   The quadrature is actually sqrt(2) times the quadrature points. This is as we are integrating
      //      against the normal distribution, not exp(-x^2)
      const PetscReal quad[4] = {-0.74196378430272585764851359672636022482952014750891895361147387899499975465000530,
                                 0.74196378430272585764851359672636022482952014750891895361147387899499975465000530,
                                -2.3344142183389772393175122672103621944890707102161406718291603341725665622712306,
                                 2.3344142183389772393175122672103621944890707102161406718291603341725665622712306};
      // The weights are the true weights divided by sqrt(pi)
      const PetscReal weights[4] = {0.45412414523193150818310700622549094933049562338805584403605771393758003145477625,
                                   0.45412414523193150818310700622549094933049562338805584403605771393758003145477625,
                                   0.045875854768068491816892993774509050669504376611944155963942286062419968545223748,
                                   0.045875854768068491816892993774509050669504376611944155963942286062419968545223748};
      // Factor to multiply the grid spacing by to get the standard deviation
      const PetscReal sigmaFactor = 1.0;
      // Interpolation list for fast integration
      PetscInt *interpCellList = nullptr;

      PetscInt *globalIndices = nullptr;


      std::shared_ptr<ablate::domain::rbf::RBF> vertRBF = nullptr;
      std::shared_ptr<ablate::domain::rbf::RBF> cellRBF = nullptr;

      std::shared_ptr<ablate::domain::Region> region = nullptr;
      std::shared_ptr<ablate::domain::SubDomain> subDomain = nullptr;

      DM vertDM = nullptr, vertGradDM = nullptr;  // DM for vertex-based data
      DM cellDM = nullptr, cellGradDM = nullptr;  // DM for cell-center data

      Vec vertVec[2] = {nullptr, nullptr}, vertGradVec[2] = {nullptr, nullptr};  // Vertex-based data
      Vec cellVec[2] = {nullptr, nullptr}, cellGradVec[2] = {nullptr, nullptr};  // Cell-based data

      // Store the cell and vert ranges so that they don't have to be re-computed every iteration
      ablate::domain::Range cellRange = {};
      ablate::domain::Range vertRange = {};
      ablate::domain::ReverseRange reverseVertRange = {};
      ablate::domain::ReverseRange reverseCellRange = {};

      // The cell and vertex lists where to perform calculations
      PetscInt nCellList = 0, *cellList = nullptr, *reverseCellList = nullptr;
      PetscInt nVertList = 0, *vertList = nullptr, *reverseVertList = nullptr;



    public:

      Reconstruction(const std::shared_ptr<ablate::domain::SubDomain> subDomain, std::shared_ptr<ablate::domain::Region> region = nullptr);
      ~Reconstruction();


      // Given a cell-centered VOF field compute the level-set field
      void ToLevelSet(const ablate::domain::Field vofField);

  };


}  // namespace ablate::levelSet::Reconstruction
#endif  // ABLATELIBRARY_INTERFACERECONSTRUCTION_HPP
