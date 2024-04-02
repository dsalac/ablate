#ifndef ABLATELIBRARY_INTERFACERECONSTRUCTION_HPP
#define ABLATELIBRARY_INTERFACERECONSTRUCTION_HPP

#include "domain/RBF/ga.hpp"
#include "domain/RBF/hybrid.hpp"
#include "domain/RBF/imq.hpp"
#include "domain/RBF/mq.hpp"
#include "domain/RBF/phs.hpp"
#include "domain/RBF/rbf.hpp"
#include "domain/subDomain.hpp"



#include "domain/range.hpp"
#include "domain/reverseRange.hpp"



namespace ablate::levelSet {

  class Reconstruction {

    private:



      void BuildInterpCellList(DM dm, const ablate::domain::Range cellRange);

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




      std::shared_ptr<ablate::domain::rbf::RBF> vertRBF = nullptr;
      std::shared_ptr<ablate::domain::rbf::RBF> cellRBF = nullptr;

      const std::shared_ptr<ablate::domain::SubDomain> subDomain = {};
      const ablate::domain::Range cellRange = {};

      DM vertDM;  // DM for vertex-based data
      DM cellDM;  // DM for cell-center data

      Vec lsVec_local, lsVec_global;  // Vertex-based vectors for level sets

    public:
      Reconstruction(ablate::domain::Range cellRange, const std::shared_ptr<ablate::domain::SubDomain> subDomain);
      ~Reconstruction();


      // Given a cell-centered VOF field compute the level-set field
      void ToLevelSet(const ablate::domain::Field vofField);

  };


}  // namespace ablate::levelSet::Reconstruction
#endif  // ABLATELIBRARY_INTERFACERECONSTRUCTION_HPP
