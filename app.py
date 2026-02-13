from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import os, cv2, time, traceback

from preprocessing import enhance_mri_contrast
from classification import load_classifier, classify_image, is_brain_mri
from explainable_ai import gradcam_for_resnet, overlay_cam, analyze_cam
from segmentation import (
    load_segmentation_model,
    segment_tumor,
    overlay_mask,
    compute_tumor_stats,
    draw_bounding_box
)
from xai_text import generate_dynamic_explanation
from report_generator import generate_pdf


# -------------------------------------------------
# Flask Setup
# -------------------------------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")

UPLOAD_FOLDER = "uploads"
REPORT_FOLDER = "reports"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# -------------------------------------------------
# Load Models ONCE
# -------------------------------------------------
print("🔄 Loading AI models...")

# Classification model
try:
    CLS_MODEL = load_classifier("models/weights/best_resnet50_brain_tumor.pth")
    print("✅ Classification model loaded")
except Exception as e:
    CLS_MODEL = None
    print("❌ Classifier failed:", e)

# Segmentation model
try:
    SEG_MODEL = load_segmentation_model("models/weights/best_unet_brain_tumor.pth")
    print("✅ Segmentation model loaded")
except Exception as e:
    SEG_MODEL = None
    print("❌ Segmentation failed:", e)

print("🚀 TumorDetect AI initialized")


# -------------------------------------------------
# Helper: Risk Level
# -------------------------------------------------
def compute_risk(tumor_type, confidence):
    if tumor_type.lower() in ["no tumor", "no_tumor"]:
        return "No Risk"
    if confidence >= 0.85:
        return "High Risk"
    elif confidence >= 0.65:
        return "Medium Risk"
    else:
        return "Low Risk"


# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/reports/<path:filename>")
def download_report(filename):
    return send_from_directory(REPORT_FOLDER, filename, as_attachment=True)


# -------------------------------------------------
# Upload + AI Processing
# -------------------------------------------------
@app.route("/upload", methods=["POST"])
def upload_file():

    if CLS_MODEL is None:
        return jsonify({"error": "Classifier unavailable"}), 503

    try:
        # ---------- Patient Details ----------
        patient_id = request.form.get("patient_id", "Unknown")
        patient_name = request.form.get("patient_name", "Unknown")
        age = request.form.get("age", "N/A")
        gender = request.form.get("gender", "N/A")
        mri_date = request.form.get("mri_date", "N/A")

        # ---------- File Validation ----------
        file = request.files.get("file")
        if not file or file.filename == "":
            return jsonify({"error": "No file uploaded"}), 400

        if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            return jsonify({"error": "Only PNG/JPG images allowed"}), 400

        # ---------- Save Original ----------
        ts = int(time.time())
        safe_name = file.filename.replace(" ", "_")
        base_name = f"{ts}_{safe_name}"

        orig_path = os.path.join(UPLOAD_FOLDER, base_name)
        file.save(orig_path)

        # ---------- Preprocessing ----------
        enhanced_bgr = enhance_mri_contrast(orig_path)
        enhanced_path = os.path.join(UPLOAD_FOLDER, f"enhanced_{base_name}")
        cv2.imwrite(enhanced_path, enhanced_bgr)

        # ---------- MRI Validation ----------
        if not is_brain_mri(enhanced_bgr):
            return jsonify({
                "tumor_type": "Not an MRI Scan",
                "confidence": "N/A",
                "risk_level": "N/A",
                "original_image": url_for("uploaded_file", filename=base_name),
                "enhanced_image": url_for("uploaded_file", filename=f"enhanced_{base_name}"),
                "segmentation_image": "",
                "gradcam_image": "",
                "report_pdf": ""
            })

        # ---------- Segmentation ----------
        seg_overlay_path = ""
        tumor_stats = {}

        if SEG_MODEL is not None:
            mask = segment_tumor(SEG_MODEL, enhanced_bgr)

            # compute tumor measurements
            tumor_stats = compute_tumor_stats(mask)

            # overlay mask
            seg_overlay = overlay_mask(enhanced_bgr, mask)

            # draw bounding box
            seg_overlay = draw_bounding_box(seg_overlay, tumor_stats)

            seg_overlay_path = os.path.join(UPLOAD_FOLDER, f"seg_{base_name}")
            cv2.imwrite(seg_overlay_path, seg_overlay)

        # ---------- Classification ----------
        tumor_type, confidence, _ = classify_image(CLS_MODEL, enhanced_bgr)
        confidence_percent = f"{confidence * 100:.2f}%"
        risk_level = compute_risk(tumor_type, confidence)

        gradcam_path = ""
        details = {}

        # ---------- Explainable AI ----------
        if tumor_type.lower() not in ["no tumor", "no_tumor"]:
            cam = gradcam_for_resnet(CLS_MODEL, enhanced_bgr)
            cam_overlay = overlay_cam(enhanced_bgr, cam)

            gradcam_path = os.path.join(UPLOAD_FOLDER, f"gradcam_{base_name}")
            cv2.imwrite(gradcam_path, cam_overlay)

            cam_info = analyze_cam(cam)
            details = generate_dynamic_explanation(tumor_type, cam_info)
        else:
            details = {
                "desc": "No abnormal tumor-related activation detected.",
                "cause": "Normal brain MRI pattern.",
                "treat": "No medical intervention required."
            }

        # ---------- PDF REPORT ----------
        report_filename = f"report_{patient_id}_{ts}.pdf"
        report_path = os.path.join(REPORT_FOLDER, report_filename)

        patient_data = {
    "Patient ID": patient_id,
    "Patient Name": patient_name,
    "Age": age,
    "Gender": gender,
    "MRI Date": mri_date,

    "Tumor Type": tumor_type,
    "Risk Level": risk_level,
    "Model Confidence": confidence_percent,

    # Tumor Measurements (FIXED)
    "Tumor Location": str(tumor_stats.get("location", "N/A")),
    "Tumor Width": f"{tumor_stats.get('width_cm', 0)} cm",
    "Tumor Height": f"{tumor_stats.get('height_cm', 0)} cm",
    "Tumor Area": f"{tumor_stats.get('area_cm2', 0)} cm²",

    # AI Explanation
    "Description": details.get("desc", "N/A"),
    "Possible Cause": details.get("cause", "N/A"),
    "Treatment": details.get("treat", "N/A")
}



        image_paths = {
            "Original MRI": orig_path,
            "Enhanced MRI": enhanced_path
        }

        if seg_overlay_path:
            image_paths["Tumor Segmentation"] = seg_overlay_path

        if gradcam_path:
            image_paths["Grad-CAM Visualization"] = gradcam_path

        generate_pdf(report_path, patient_data, image_paths)

        # ---------- Final Response ----------
        return jsonify({
            "tumor_type": tumor_type,
            "confidence": confidence_percent,
            "risk_level": risk_level,
            "original_image": url_for("uploaded_file", filename=base_name),
            "enhanced_image": url_for("uploaded_file", filename=f"enhanced_{base_name}"),
            "segmentation_image": (
                url_for("uploaded_file", filename=f"seg_{base_name}")
                if seg_overlay_path else ""
            ),
            "gradcam_image": (
                url_for("uploaded_file", filename=f"gradcam_{base_name}")
                if gradcam_path else ""
            ),

            # Tumor measurements (cm)
            "tumor_location": tumor_stats.get("location"),
            "tumor_width_cm": tumor_stats.get("width_cm"),
            "tumor_height_cm": tumor_stats.get("height_cm"),
            "tumor_area_cm2": tumor_stats.get("area_cm2"),

            "tumor_description": details.get("desc", "N/A"),
            "tumor_cause": details.get("cause", "N/A"),
            "tumor_treatment": details.get("treat", "N/A"),
            "report_pdf": url_for("download_report", filename=report_filename)
        })

    except Exception:
        print(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500


# -------------------------------------------------
# Run
# -------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

