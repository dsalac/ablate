#include "boxMeshBoundaryCells.hpp"
#include <stdexcept>
#include <utility>
#include "domain/modifiers/createLabel.hpp"
#include "domain/modifiers/distributeWithGhostCells.hpp"
#include "domain/modifiers/mergeLabels.hpp"
#include "domain/modifiers/tagLabelBoundary.hpp"
#include "mathFunctions/geom/box.hpp"
#include "utilities/petscUtilities.hpp"
#include "utilities/petscSupport.hpp"

ablate::domain::BoxMeshBoundaryCells::BoxMeshBoundaryCells(const std::string& name, const std::vector<std::shared_ptr<FieldDescriptor>>& fieldDescriptors,
                                                           std::vector<std::shared_ptr<modifiers::Modifier>> preModifiers, std::vector<std::shared_ptr<modifiers::Modifier>> postModifiers,
                                                           std::vector<int> faces, const std::vector<double>& lower, const std::vector<double>& upper,
                                                           std::vector<std::string> boundary,
                                                           const std::shared_ptr<parameters::Parameters>& options,
                                                           const bool includeCorners)
    : Domain(CreateBoxDM(name, std::move(faces), lower, upper, boundary, false), name, fieldDescriptors,
             // We need to get the optional dm_plex_scale to determine bounds
             AddBoundaryModifiers(lower, upper, boundary, options ? options->Get("dm_plex_scale", 1.0) : 1.0, std::move(preModifiers), std::move(postModifiers), includeCorners), options) {
    // make sure that dm_refine was not set
    if (options) {
        if (options->Get("dm_refine", 0) != 0) {
            throw std::invalid_argument("dm_refine when used with ablate::domain::BoxMeshBoundaryCells must be 0.");
        }
    }

    // make sure that all fields have a region if we might have unused corners (dim > 1)
    if (GetDimensions() > 1) {
        for (const auto& fieldDescriptor : fieldDescriptors) {
            for (auto& fieldDescription : fieldDescriptor->GetFields()) {
                if (fieldDescription->region == nullptr) {
                    throw std::invalid_argument("All fields in ablate::domain::BoxMeshBoundaryCells::BoxMeshBoundaryCells should specify a region.");
                }
            }
        }
    }
}

ablate::domain::BoxMeshBoundaryCells::~BoxMeshBoundaryCells() {
    if (dm) {
        DMDestroy(&dm);
    }
}


void ProcessBoundaryInformation(std::size_t dim, std::vector<std::string> boundary, std::vector<DMBoundaryType> &boundaryTypes) {
    if (boundary.size() > 0) {
      for (std::size_t d = 0; d < PetscMin(dim, boundary.size()); d++) {
          PetscBool found;
          DMBoundaryType index;
          PetscEnumFind(DMBoundaryTypes, boundary[d].c_str(), (PetscEnum *)&index, &found) >> ablate::utilities::PetscUtilities::checkError;

          if (found) {
              boundaryTypes[d] = index;
          } else {
              throw std::invalid_argument("unable to find boundary type " + boundary[d]);
          }
      }
    }
    else { // Default to old behavior: all NONE
      for (std::size_t d = 0; d < PetscMin(dim, boundary.size()); d++) {
        boundaryTypes[d] = DM_BOUNDARY_NONE;
      }
    }

}

void ExpandDomain(std::size_t dim, std::vector<DMBoundaryType> boundaryTypes, std::vector<int> &faces, std::vector<double> &lower, std::vector<double> &upper) {
    for (std::size_t d = 0; d < dim; d++) {
      if (boundaryTypes[d] == DM_BOUNDARY_NONE) {
        double dx = (upper[d] - lower[d]) / faces[d];
        faces[d] += 2;
        lower[d] -= dx;
        upper[d] += dx;
      }
    }

}

DM ablate::domain::BoxMeshBoundaryCells::CreateBoxDM(const std::string& name, std::vector<int> faces, std::vector<double> lower, std::vector<double> upper, std::vector<std::string> boundary, bool simplex) {
    std::size_t dimensions = faces.size();
    if ((dimensions != lower.size()) || (dimensions != upper.size())) {
        throw std::runtime_error("BoxMesh Error: The faces, lower, and upper vectors must all be the same dimension.");
    }


    std::vector<DMBoundaryType> boundaryTypes(dimensions, DM_BOUNDARY_NONE);
    ProcessBoundaryInformation(dimensions, boundary, boundaryTypes);
    ExpandDomain(dimensions, boundaryTypes, faces, lower, upper);

    // Make copy with PetscInt
    std::vector<PetscInt> facesPetsc(faces.begin(), faces.end());
    DM dm;
    DMPlexCreateBoxMesh(PETSC_COMM_WORLD, (PetscInt)dimensions, simplex ? PETSC_TRUE : PETSC_FALSE, &facesPetsc[0], &lower[0], &upper[0], &boundaryTypes[0], PETSC_TRUE, (PetscInt)dimensions, PETSC_TRUE, &dm) >>
        utilities::PetscUtilities::checkError;

    PetscObjectSetName((PetscObject)dm, name.c_str()) >> utilities::PetscUtilities::checkError;
    return dm;
}

std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>> ablate::domain::BoxMeshBoundaryCells::AddBoundaryModifiers(std::vector<double> lower, std::vector<double> upper, std::vector<std::string> boundary, double scaleFactor,
                                                                                                                             std::vector<std::shared_ptr<modifiers::Modifier>> preModifiers,
                                                                                                                             std::vector<std::shared_ptr<modifiers::Modifier>> postModifiers, bool includeCorners) {
    // scale the bounds by the scale factor incase petsc scaled them
    for (auto& pt : lower) {
        pt *= scaleFactor;
    }
    for (auto& pt : upper) {
        pt *= scaleFactor;
    }

    auto modifiers = std::move(preModifiers);
    auto interiorLabel = std::make_shared<domain::Region>(interiorCellsLabel);
    auto boundaryFaceRegion = std::make_shared<domain::Region>(boundaryFacesLabel);
    modifiers.push_back(std::make_shared<ablate::domain::modifiers::CreateLabel>(interiorLabel, std::make_shared<ablate::mathFunctions::geom::Box>(lower, upper)));

    const int X = 0;
    const int Y = 1;
    const int Z = 2;
    const double min = std::numeric_limits<double>::lowest();
    const double max = std::numeric_limits<double>::max();

    // define a boundaryCellRegion
    modifiers.push_back(std::make_shared<ablate::domain::modifiers::TagLabelBoundary>(interiorLabel, boundaryFaceRegion));

    // preDefineAllBoundaryRegions
    auto boundaryCellsFrontRegion = std::make_shared<domain::Region>(boundaryCellsFront);
    auto boundaryCellsBackRegion = std::make_shared<domain::Region>(boundaryCellsBack);
    auto boundaryCellsTopRegion = std::make_shared<domain::Region>(boundaryCellsTop);
    auto boundaryCellsBottomRegion = std::make_shared<domain::Region>(boundaryCellsBottom);
    auto boundaryCellsRightRegion = std::make_shared<domain::Region>(boundaryCellsRight);
    auto boundaryCellsLeftRegion = std::make_shared<domain::Region>(boundaryCellsLeft);


    PetscInt dim = lower.size();
    std::vector<DMBoundaryType> boundaryTypes(dim, DM_BOUNDARY_NONE);
    ProcessBoundaryInformation(dim, boundary, boundaryTypes);

    double cornerShift = (includeCorners ? max : 0);

    // Define a subset for the other boundary regions
    std::vector<std::shared_ptr<domain::Region>> boundaryRegions;
    switch (dim) {
        case 3:
            if (boundaryTypes[Z]==DM_BOUNDARY_NONE) {
                modifiers.push_back(std::make_shared<ablate::domain::modifiers::CreateLabel>(
                    boundaryCellsFrontRegion, std::make_shared<ablate::mathFunctions::geom::Box>(std::vector<double>{lower[X] - cornerShift, lower[Y] - cornerShift, upper[Z]}, std::vector<double>{upper[X] + cornerShift, upper[Y] + cornerShift, max})));
                boundaryRegions.push_back(boundaryCellsFrontRegion);

                modifiers.push_back(std::make_shared<ablate::domain::modifiers::CreateLabel>(
                    boundaryCellsBackRegion, std::make_shared<ablate::mathFunctions::geom::Box>(std::vector<double>{lower[X] - cornerShift, lower[Y] - cornerShift, min}, std::vector<double>{upper[X] + cornerShift, upper[Y] + cornerShift, lower[Z]})));
                boundaryRegions.push_back(boundaryCellsBackRegion);
            }

            if (boundaryTypes[Y]==DM_BOUNDARY_NONE) {
                modifiers.push_back(std::make_shared<ablate::domain::modifiers::CreateLabel>(
                    boundaryCellsTopRegion, std::make_shared<ablate::mathFunctions::geom::Box>(std::vector<double>{lower[X] - cornerShift, upper[Y], lower[Z] - cornerShift}, std::vector<double>{upper[X] + cornerShift, max, upper[Z] + cornerShift})));
                boundaryRegions.push_back(boundaryCellsTopRegion);

                modifiers.push_back(std::make_shared<ablate::domain::modifiers::CreateLabel>(
                    boundaryCellsBottomRegion, std::make_shared<ablate::mathFunctions::geom::Box>(std::vector<double>{lower[X] - cornerShift, min, lower[Z] - cornerShift}, std::vector<double>{upper[X] + cornerShift, lower[Y], upper[Z] + cornerShift})));
                boundaryRegions.push_back(boundaryCellsBottomRegion);
            }

            if (boundaryTypes[X]==DM_BOUNDARY_NONE) {
                modifiers.push_back(std::make_shared<ablate::domain::modifiers::CreateLabel>(
                    boundaryCellsRightRegion, std::make_shared<ablate::mathFunctions::geom::Box>(std::vector<double>{upper[X], lower[Y] - cornerShift, lower[Z] - cornerShift}, std::vector<double>{max, upper[Y] + cornerShift, upper[Z] + cornerShift})));
                boundaryRegions.push_back(boundaryCellsRightRegion);

                modifiers.push_back(std::make_shared<ablate::domain::modifiers::CreateLabel>(
                    boundaryCellsLeftRegion, std::make_shared<ablate::mathFunctions::geom::Box>(std::vector<double>{min, lower[Y] - cornerShift, lower[Z] - cornerShift}, std::vector<double>{lower[X], upper[Y] + cornerShift, upper[Z] + cornerShift})));
                boundaryRegions.push_back(boundaryCellsLeftRegion);
            }
            break;
        case 2:
            if (boundaryTypes[Y]==DM_BOUNDARY_NONE) {
                modifiers.push_back(std::make_shared<ablate::domain::modifiers::CreateLabel>(
                    boundaryCellsTopRegion, std::make_shared<ablate::mathFunctions::geom::Box>(std::vector<double>{lower[X] - cornerShift, upper[Y]}, std::vector<double>{upper[X] + cornerShift, max})));
                boundaryRegions.push_back(boundaryCellsTopRegion);

                modifiers.push_back(std::make_shared<ablate::domain::modifiers::CreateLabel>(
                    boundaryCellsBottomRegion, std::make_shared<ablate::mathFunctions::geom::Box>(std::vector<double>{lower[X] - cornerShift, min}, std::vector<double>{upper[X] + cornerShift, lower[Y]})));
                boundaryRegions.push_back(boundaryCellsBottomRegion);
            }

            if (boundaryTypes[X]==DM_BOUNDARY_NONE) {
                modifiers.push_back(std::make_shared<ablate::domain::modifiers::CreateLabel>(
                    boundaryCellsRightRegion, std::make_shared<ablate::mathFunctions::geom::Box>(std::vector<double>{upper[X], lower[Y] - cornerShift}, std::vector<double>{max, upper[Y] + cornerShift})));
                boundaryRegions.push_back(boundaryCellsRightRegion);

                modifiers.push_back(std::make_shared<ablate::domain::modifiers::CreateLabel>(
                    boundaryCellsLeftRegion, std::make_shared<ablate::mathFunctions::geom::Box>(std::vector<double>{min, lower[Y] - cornerShift}, std::vector<double>{lower[X], upper[Y] + cornerShift})));
                boundaryRegions.push_back(boundaryCellsLeftRegion);
            }

            break;
        case 1:
            if (boundaryTypes[X]==DM_BOUNDARY_NONE) {
                modifiers.push_back(std::make_shared<ablate::domain::modifiers::CreateLabel>(boundaryCellsRightRegion,
                                                                                             std::make_shared<ablate::mathFunctions::geom::Box>(std::vector<double>{upper[X]}, std::vector<double>{max})));
                boundaryRegions.push_back(boundaryCellsRightRegion);

                modifiers.push_back(std::make_shared<ablate::domain::modifiers::CreateLabel>(boundaryCellsLeftRegion,
                                                                                             std::make_shared<ablate::mathFunctions::geom::Box>(std::vector<double>{min}, std::vector<double>{lower[X]})));
                boundaryRegions.push_back(boundaryCellsLeftRegion);
            }
    }

    // define the boundaryCellRegion
    auto boundaryCellRegion = std::make_shared<domain::Region>(boundaryCellsLabel);
    modifiers.push_back(std::make_shared<ablate::domain::modifiers::MergeLabels>(boundaryCellRegion, boundaryRegions));

    // define the ghost cells plus interior
    auto entireDomainRegion = std::make_shared<domain::Region>(entireDomainLabel);
    modifiers.push_back(std::make_shared<ablate::domain::modifiers::MergeLabels>(entireDomainRegion, std::vector<std::shared_ptr<domain::Region>>{interiorLabel, boundaryCellRegion}));

    modifiers.insert(modifiers.end(), postModifiers.begin(), postModifiers.end());

    return modifiers;
}

#include "registrar.hpp"
REGISTER(ablate::domain::Domain, ablate::domain::BoxMeshBoundaryCells,
         "simple uniform box mesh with boundary solver cells.  Available labels are: interiorCells, domain (interior and boundary cells), boundaryFaces, boundaryCells, boundaryCellsLeft, "
         "boundaryCellsRight, "
         "boundaryCellsBottom, boundaryCellsTop, boundaryCellsFront, and boundaryCellsBack",
         ARG(std::string, "name", "the name of the domain/mesh object"), OPT(std::vector<ablate::domain::FieldDescriptor>, "fields", "a list of fields/field descriptors"),
         OPT(std::vector<ablate::domain::modifiers::Modifier>, "preModifiers", "a list of domain modifiers to apply before ghost labeling"),
         OPT(std::vector<ablate::domain::modifiers::Modifier>, "postModifiers", "a list of domain modifiers to apply after ghost labeling"),
         ARG(std::vector<int>, "faces", "the number of faces in each direction"), ARG(std::vector<double>, "lower", "the lower bound of the mesh"),
         ARG(std::vector<double>, "upper", "the upper bound of the mesh"),
         OPT(std::vector<std::string>, "boundary", "custom boundary types (NONE, GHOSTED, MIRROR, PERIODIC)"),
         OPT(ablate::parameters::Parameters, "options", "PETSc options specific to this dm.  Default value allows the dm to access global options."),
         OPT(bool, "includeCorners", "Include corner cells when labelling the boundary cells. Default is false."));
