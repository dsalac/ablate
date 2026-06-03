#ifndef ABLATELIBRARY_NPHASESURFACETENSION_HPP
#define ABLATELIBRARY_NPHASESURFACETENSION_HPP

#include <petsc.h>
#include <memory>
#include <vector>
#include "domain/range.hpp"
#include "domain/RBF/rbf.hpp"
#include "domain/reverseRange.hpp"
#include "finiteVolume/fluxCalculator/fluxCalculator.hpp"
#include "flowProcess.hpp"
#include "process.hpp"
#include "solver/solver.hpp"
#include "finiteVolume/stencils/gaussianConvolution.hpp"
#include "finiteVolume/nPhaseFlowFields.hpp"

namespace ablate::finiteVolume::processes {

class NPhaseSurfaceTension : public Process {
   private:

    std::shared_ptr<ablate::domain::SubDomain> subDomain;

    std::vector<PetscReal> sigmaij;
    std::size_t nPhases;
    PetscReal h;


    static PetscErrorCode PointFlux(PetscInt dim, const PetscFVFaceGeom* fg,
  const PetscInt uOff[], const PetscInt uOff_x[],
  const PetscScalar fieldL[], const PetscScalar fieldR[], const PetscScalar field[], const PetscScalar grad[],
  const PetscInt aOff[], const PetscInt aOff_x[],
  const PetscScalar auxL[], const PetscScalar auxR[], const PetscScalar aux[], const PetscScalar gradAux[],
  PetscScalar flux[], void* ctx);

   public:
    explicit NPhaseSurfaceTension(const std::vector<PetscReal>& sigmaij);

    void Setup(ablate::finiteVolume::FiniteVolumeSolver& flow) override;
    void Initialize(ablate::finiteVolume::FiniteVolumeSolver& flow) override;
};

}  // namespace ablate::finiteVolume::processes

#endif


