#ifndef ABLATELIBRARY_GRAVITY_HPP
#define ABLATELIBRARY_GRAVITY_HPP

#include "flowProcess.hpp"

namespace ablate::finiteVolume::processes {

class Gravity : public FlowProcess {
   private:
    const std::vector<PetscReal> gravityVector;
    const std::string fieldType;
    const PetscReal rho0;

    /**
     * private function to compute gravity source
     * @return
     */
    static PetscErrorCode ComputeGravitySourceEuler(PetscInt dim, PetscReal time, const PetscFVCellGeom* cg, const PetscInt uOff[], const PetscScalar u[], const PetscInt aOff[], const PetscScalar a[], PetscScalar f[], void* ctx);

    static PetscErrorCode ComputeGravitySourceAllaire(PetscInt dim, PetscReal time, const PetscFVCellGeom* cg, const PetscInt uOff[], const PetscScalar u[], const PetscInt aOff[], const PetscScalar a[], PetscScalar f[], void* ctx);




   public:
    explicit Gravity(std::vector<double> gravityVector, const std::string& fieldType, const PetscReal rho0 = 0);
    void Setup(ablate::finiteVolume::FiniteVolumeSolver& fv) override;
};

}  // namespace ablate::finiteVolume::processes
#endif  // ABLATELIBRARY_GRAVITY_HPP
