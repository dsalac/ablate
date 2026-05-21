#include "gravity.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/nPhaseFlowFields.hpp"
#include "finiteVolume/processes/nPhaseAllaireAdvection.hpp"

ablate::finiteVolume::processes::Gravity::Gravity(std::vector<double> gravityVector, const std::string& fieldType, const PetscReal rho0) :
    gravityVector(gravityVector),
    fieldType(fieldType.empty() ? "CompressibleFlowFields" : fieldType),
    rho0(rho0) {}

void ablate::finiteVolume::processes::Gravity::Setup(ablate::finiteVolume::FiniteVolumeSolver &fv) {
    // add the source function

    if (fieldType == "CompressibleFlowFields") {
      fv.RegisterRHSFunction(ComputeGravitySourceEuler, this, {CompressibleFlowFields::EULER_FIELD}, {CompressibleFlowFields::EULER_FIELD}, {});
    }
    else if (fieldType == "NPhaseFlowFields") {
      std::vector<std::string> requiredFieldList;
      std::vector<ablate::domain::FieldLocation> requiredLocationList;

      requiredFieldList.push_back(ablate::finiteVolume::NPhaseFlowFields::ALLAIRE);
      requiredLocationList.push_back(ablate::domain::FieldLocation::SOL);

      requiredFieldList.push_back(ablate::finiteVolume::NPhaseFlowFields::UI);
      requiredLocationList.push_back(ablate::domain::FieldLocation::AUX);

      requiredFieldList.push_back(ablate::finiteVolume::NPhaseFlowFields::RHO);
      requiredLocationList.push_back(ablate::domain::FieldLocation::AUX);

      std::size_t k = 0;
      for (auto fieldName : requiredFieldList) {
        if (!(fv.GetSubDomain().ContainsField(fieldName))) {
          throw std::runtime_error("ablate::finiteVolume::processes::IntSharp expects a "+ fieldName +" field to be defined.");
        }
        const ablate::domain::Field field = fv.GetSubDomain().GetField(fieldName);
        if (field.location != requiredLocationList[k++]) {
          throw std::runtime_error("ablate::finiteVolume::processes::IntSharp: "+ fieldName +" is in the incorrect location.");
        }
      }

      int advLoc = fv.FindProcessLocation<ablate::finiteVolume::processes::NPhaseAllaireAdvection>();
      int gravityLoc = fv.FindProcessLocation<ablate::finiteVolume::processes::Gravity>();

      if (advLoc < 0 || advLoc > gravityLoc) throw std::runtime_error("The process ablate::finiteVolume::processes::NPhaseAllaireAdvection must be before ablate::finiteVolume::processes::Gravity");

      fv.RegisterRHSFunction(ComputeGravitySourceAllaire, this, {NPhaseFlowFields::ALLAIRE}, {}, {NPhaseFlowFields::RHO, NPhaseFlowFields::UI});

    }
    else throw std::invalid_argument("Unknown flow field of " + fieldType + " in ablate::finiteVolume::processes::Gravity");
}


PetscErrorCode ablate::finiteVolume::processes::Gravity::ComputeGravitySourceAllaire(PetscInt dim, PetscReal time, const PetscFVCellGeom *cg,
          const PetscInt *uOff, const PetscScalar *u,
          const PetscInt *aOff, const PetscScalar *a,
          PetscScalar *f, void *ctx) {
    PetscFunctionBeginUser;

    auto gravityProcess = (ablate::finiteVolume::processes::Gravity *)ctx;

    const PetscReal density = a[aOff[0]] - gravityProcess->rho0;

    // Add in the gravity source terms for momentum and energy
    f[NPhaseFlowFields::RHOE] = 0;
    for (PetscInt d = 0; d < dim; d++) {
        f[NPhaseFlowFields::RHOU + d] = density * gravityProcess->gravityVector[d];
        f[NPhaseFlowFields::RHOE] += a[aOff[1] + d] * f[NPhaseFlowFields::RHOU + d];
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode ablate::finiteVolume::processes::Gravity::ComputeGravitySourceEuler(PetscInt dim, PetscReal time, const PetscFVCellGeom *cg, const PetscInt *uOff, const PetscScalar *u, const PetscInt *aOff,
                                                                              const PetscScalar *a, PetscScalar *f, void *ctx) {
    PetscFunctionBeginUser;
    const int EULER_FIELD = 0;
    auto gravityProcess = (ablate::finiteVolume::processes::Gravity *)ctx;

    // exact some values
    const PetscReal density = u[uOff[EULER_FIELD] + CompressibleFlowFields::RHO];

    // set the source terms
    f[CompressibleFlowFields::RHO] = 0.0;
    f[CompressibleFlowFields::RHOE] = 0.0;

    // Add in the gravity source terms for momentum and energy
    for (PetscInt n = 0; n < dim; n++) {
        f[CompressibleFlowFields::RHOU + n] = density * gravityProcess->gravityVector[n];
        PetscReal vel = u[uOff[EULER_FIELD] + CompressibleFlowFields::RHOU + n] / density;
        f[CompressibleFlowFields::RHOE] += vel * f[CompressibleFlowFields::RHOU + n];
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::Gravity, "build advection/diffusion for the euler field",
         ARG(std::vector<double>, "vector", "gravitational acceleration vector"),
         OPT(std::string, "fieldType", "type of flow field this is being applied to. Default is CompressibleFlowFields"),
         OPT(PetscReal, "rho0", "background density to subtract. Default is 0")
         );
