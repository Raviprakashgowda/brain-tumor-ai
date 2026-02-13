def generate_dynamic_explanation(tumor_type, cam_info):
    """
    Generates image-specific explanation using Grad-CAM properties
    """
    area = cam_info.get("activated_area", 0)
    strength = cam_info.get("mean_activation", 0)

    # ---------------- PITUITARY ----------------
    if tumor_type == "Pituitary":
        if area < 0.15:
            return {
                "desc": (
                    "Grad-CAM highlights a small, focused activation near the pituitary gland, "
                    "indicating a localized abnormality in this scan."
                ),
                "cause": (
                    "Such localized activation is commonly associated with benign pituitary adenomas."
                ),
                "treat": (
                    "These tumors are often managed using medication or minimally invasive surgery."
                )
            }
        else:
            return {
                "desc": (
                    "Grad-CAM shows a broader activation around the pituitary region, "
                    "suggesting larger tissue involvement."
                ),
                "cause": (
                    "Wider activation may indicate a larger pituitary lesion affecting hormonal regulation."
                ),
                "treat": (
                    "Surgical intervention followed by hormonal therapy may be required."
                )
            }

    # ---------------- GLIOMA ----------------
    elif tumor_type == "Glioma":
        if strength > 0.4:
            return {
                "desc": (
                    "Grad-CAM highlights diffuse and strong activation across brain tissue, "
                    "indicating infiltrative tumor characteristics."
                ),
                "cause": (
                    "This pattern suggests aggressive cellular growth typically seen in high-grade gliomas."
                ),
                "treat": (
                    "Treatment usually involves surgery combined with radiotherapy and chemotherapy."
                )
            }
        else:
            return {
                "desc": (
                    "Moderate Grad-CAM activation suggests a more localized tumor pattern."
                ),
                "cause": (
                    "This may correspond to a lower-grade glioma with slower growth."
                ),
                "treat": (
                    "Surgical removal followed by monitoring is commonly recommended."
                )
            }

    # ---------------- MENINGIOMA ----------------
    elif tumor_type == "Meningioma":
        return {
            "desc": (
                "Grad-CAM highlights a well-defined activation near the brain surface, "
                "which is characteristic of meningioma tumors."
            ),
            "cause": (
                "Meningiomas arise from the meninges and are often slow-growing and benign."
            ),
            "treat": (
                "Observation or surgical removal is typically sufficient depending on size and symptoms."
            )
        }

    # ---------------- NO TUMOR ----------------
    elif tumor_type == "No Tumor":
        return {
            "desc": (
                "Grad-CAM does not show any concentrated activation regions, "
                "indicating the absence of abnormal tumor patterns in this scan."
            ),
            "cause": (
                "The MRI scan appears within normal limits."
            ),
            "treat": (
                "No medical treatment is required."
            )
        }

    # ---------------- FALLBACK ----------------
    return {
        "desc": "AI analysis completed with limited explainability.",
        "cause": "Insufficient activation patterns detected.",
        "treat": "Clinical evaluation recommended."
    }
