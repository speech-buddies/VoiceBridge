# Software Requirements Specification (Volere template)

Source for the **Software Requirements Specification (SRS)** using the **Volere-inspired** 

| File | Role |
|------|------|
| `SRS.tex` | Main SRS document: goals, constraints, functional and non-functional requirements, and traceability hooks to other artifacts. |

This folder is intended to pair with `docs/Common.tex` and `docs/Comments.tex` via `\input` paths. Risk sections may be omitted where hazards are covered in `../HazardAnalysis/`, per template notes inside the file.

If your course Makefile expects a folder literally named `SRS`, rename or adjust build scripts accordingly.
