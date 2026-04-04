# Design documentation

This folder holds **software design** artifacts for VoiceBridge: high-level structure (module guide) and detailed module interfaces. Sources are LaTeX; build with each subfolder’s `Makefile` where present.

| Subfolder | Contents |
|-----------|----------|
| [SoftArchitecture](SoftArchitecture/) | **Module Guide (MG)** — architecture, decomposition, uses hierarchy, and traceability to requirements (`MG.tex`). |
| [SoftDetailedDes](SoftDetailedDes/) | **Module Interface Specification (MIS)** — formal interfaces, semantics, and module-level detail (`MIS.tex`). |

Shared macros and comments are pulled from the parent `docs/` tree (`Common.tex`, `Comments.tex`). Figures may reference `docs/imgs/`.
