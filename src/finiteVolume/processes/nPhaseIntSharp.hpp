#ifndef ABLATELIBRARY_FINITEVOLUME_INTSHARP_HPP
#define ABLATELIBRARY_FINITEVOLUME_INTSHARP_HPP

#include <petsc.h>
#include <memory>
#include <string>
#include <vector>
#include "domain/range.hpp"
#include "finiteVolume/fluxCalculator/fluxCalculator.hpp"
#include "flowProcess.hpp"
#include "process.hpp"
#include "solver/solver.hpp"
#include "nPhaseAllaireAdvection.hpp"
#include "eos/kthStiffenedGas.hpp"

namespace ablate::finiteVolume::processes {

class NPhaseIntSharp : public Process {

   private:
    const PetscReal Gamma;
    const PetscReal epsilon;
    PetscReal h;
    const PetscReal p0; // Initial pressure
    const PetscInt preGauss = 0;
    const PetscInt postGauss = 0;

    PetscBool preStageHasRun = PETSC_FALSE;

    std::vector<std::shared_ptr<ablate::eos::KthStiffenedGas>> eosNPhase;
    std::shared_ptr<ablate::domain::SubDomain> subDomain;

    static PetscErrorCode NPhaseIntSharpPointFlux(PetscInt dim, const PetscFVFaceGeom* fg,
      const PetscInt uOff[], const PetscInt uOff_x[],
      const PetscScalar fieldL[], const PetscScalar fieldR[], const PetscScalar field[], const PetscScalar grad[],
      const PetscInt aOff[], const PetscInt aOff_x[],
      const PetscScalar auxL[], const PetscScalar auxR[], const PetscScalar aux[], const PetscScalar gradAux[],
      PetscScalar flux[], void* ctx);

    PetscErrorCode AdvectionFlux(
      PetscInt dim, const PetscFVFaceGeom* fg,
      const PetscInt uOff[], const PetscScalar fieldL[], const PetscScalar fieldR[],
      const PetscInt aOff[], const PetscScalar auxL[], const PetscScalar auxR[],
      PetscScalar flux[], void* ctx);

    static PetscErrorCode SharpeningFlux(PetscInt dim, const PetscFVFaceGeom* fg,
      const PetscInt uOff[], const PetscInt uOff_x[],
      const PetscScalar fieldL[], const PetscScalar fieldR[], const PetscScalar field[], const PetscScalar grad[],
      const PetscInt aOff[], const PetscInt aOff_x[],
      const PetscScalar auxL[], const PetscScalar auxR[], const PetscScalar aux[], const PetscScalar gradAux[],
      PetscScalar flux[], void* ctx);

  static PetscErrorCode SharpeningFluxAllFields(PetscInt dim, const PetscFVFaceGeom* fg,
      const PetscInt uOff[], const PetscInt uOff_x[],
      const PetscScalar fieldL[], const PetscScalar fieldR[], const PetscScalar field[], const PetscScalar grad[],
      const PetscInt aOff[], const PetscInt aOff_x[],
      const PetscScalar auxL[], const PetscScalar auxR[], const PetscScalar aux[], const PetscScalar gradAux[],
      PetscScalar flux[], void* ctx);

   public:

    PetscErrorCode NPhaseIntSharpPreSharp(TS flowTS, ablate::solver::Solver &solver);

    explicit NPhaseIntSharp(
        const PetscReal Gamma,
        const PetscReal epsilon,
        const PetscReal p0,
        const PetscInt preGauss,
        const PetscInt postGauss
      );

    ~NPhaseIntSharp() override;

    void Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) override;
    void Initialize(ablate::finiteVolume::FiniteVolumeSolver &flow) override;


};
}  // namespace ablate::finiteVolume::processes
#endif
