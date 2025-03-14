#ifndef ABLATELIBRARY_FINITEVOLUME_INTSHARP_HPP
#define ABLATELIBRARY_FINITEVOLUME_INTSHARP_HPP

#include <petsc.h>
#include <memory>
#include <vector>
#include "domain/range.hpp"
#include "finiteVolume/fluxCalculator/fluxCalculator.hpp"
#include "flowProcess.hpp"
#include "process.hpp"
#include "solver/solver.hpp"
#include "twoPhaseEulerAdvection.hpp"
#include "finiteVolume/stencils/gaussianConvolution.hpp"


namespace ablate::finiteVolume::processes {

class IntSharp : public Process, public ablate::utilities::Loggable<IntSharp> {


   private:
    //coeffs
    PetscReal Gamma;
    PetscReal epsilon;
    DM cellDM = nullptr;
    DM cellGradDM = nullptr;
    std::shared_ptr<ablate::finiteVolume::stencil::GaussianConvolution> faceGaussianConv = nullptr;
    std::shared_ptr<ablate::finiteVolume::stencil::GaussianConvolution> cellGaussianConv = nullptr;
    enum VecLoc { LOCAL , GLOBAL };
    const PetscReal phiRange[2] = {1.e-4, 1.0 - 1.e-4};

    void ClearData();

    void SetMasks();

    struct vecData {
      DM dm;
      Vec vec;
      PetscScalar *array;
    };

    std::vector<struct vecData> localVecList = {};
    std::vector<struct vecData> globalVecList = {};

    void MemoryHelper(DM dm, VecLoc loc, Vec *vec, PetscScalar **array);
    void MemoryHelper();

    PetscErrorCode IntSharpPreStage(TS flowTs, ablate::solver::Solver &flow, PetscReal stagetime);
    PetscErrorCode IntSharpPreStep(TS flowTs, ablate::solver::Solver &flow);\

    void SetMask(ablate::domain::Range &cellRange, DM phiDM, Vec phiVec, PetscInt *faceMask);
    void CopyVOFData(TS ts, const ablate::domain::SubDomain& subDomain, ablate::domain::Range cellRange, DM cellDM, Vec phiVec[2], PetscScalar *phiArray[2]);
    void UpdateSolVec(TS ts, ablate::domain::SubDomain& subDomain, ablate::domain::Range cellRange, DM cellDM, PetscScalar *newPhiArray);



   public:
    /**
     *
     * @param Gamma
     * @param epsilon
     */
    explicit IntSharp(PetscReal epsilon);

    /**
     * Clean up the dm created
     */
    ~IntSharp() override;

    /**
     * Setup the process to define the vertex dm
     * @param flow
     */
    void Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) override;
    void Initialize(ablate::finiteVolume::FiniteVolumeSolver &flow) override;

};
}  // namespace ablate::finiteVolume::processes
#endif
