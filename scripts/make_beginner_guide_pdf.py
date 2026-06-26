from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Flowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "beginner_guide_ch4_synthetic_publication.pdf"


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.strip().replace("\n", " "), style)


def bullets(items: list[str], style: ParagraphStyle) -> ListFlowable:
    return ListFlowable(
        [ListItem(p(item, style), bulletColor=colors.HexColor("#2F5597")) for item in items],
        bulletType="bullet",
        leftIndent=18,
        bulletFontName="Helvetica",
        bulletFontSize=8,
    )


class FlowDiagram(Flowable):
    def __init__(self, labels: list[str], width: float = 7.0 * inch, height: float = 1.05 * inch):
        super().__init__()
        self.labels = labels
        self.width = width
        self.height = height

    def draw(self):
        c = self.canv
        n = len(self.labels)
        gap = 0.13 * inch
        box_w = (self.width - gap * (n - 1)) / n
        box_h = 0.55 * inch
        y = 0.28 * inch
        for i, label in enumerate(self.labels):
            x = i * (box_w + gap)
            c.setFillColor(colors.HexColor("#EAF2F8"))
            c.setStrokeColor(colors.HexColor("#2F5597"))
            c.roundRect(x, y, box_w, box_h, 6, fill=1, stroke=1)
            c.setFillColor(colors.HexColor("#1F1F1F"))
            c.setFont("Helvetica-Bold", 7.7)
            lines = label.split("|")
            for j, line in enumerate(lines):
                c.drawCentredString(x + box_w / 2, y + box_h / 2 + (len(lines) - 1) * 5 - j * 10 - 3, line)
            if i < n - 1:
                c.setStrokeColor(colors.HexColor("#555555"))
                c.line(x + box_w + 0.02 * inch, y + box_h / 2, x + box_w + gap - 0.02 * inch, y + box_h / 2)
                c.line(x + box_w + gap - 0.08 * inch, y + box_h / 2 + 0.04 * inch, x + box_w + gap - 0.02 * inch, y + box_h / 2)
                c.line(x + box_w + gap - 0.08 * inch, y + box_h / 2 - 0.04 * inch, x + box_w + gap - 0.02 * inch, y + box_h / 2)


def add_header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(0.72 * inch, 0.45 * inch, "CH4 Plume Synthetic Publication - Beginner Guide")
    canvas.drawRightString(7.78 * inch, 0.45 * inch, f"Page {doc.page}")
    canvas.restoreState()


def table(data: list[list[str]], widths: list[float] | None = None) -> Table:
    header_style = ParagraphStyle(
        "TableHeader",
        fontName="Helvetica-Bold",
        fontSize=8.0,
        leading=9.5,
        textColor=colors.white,
    )
    body_style = ParagraphStyle(
        "TableBody",
        fontName="Helvetica",
        fontSize=7.6,
        leading=9.4,
        textColor=colors.black,
        wordWrap="CJK",
    )
    wrapped = []
    for row_index, row in enumerate(data):
        style = header_style if row_index == 0 else body_style
        wrapped.append([
            cell if isinstance(cell, Paragraph) else Paragraph(str(cell), style)
            for cell in row
        ])
    t = Table(wrapped, colWidths=widths, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2F5597")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.2),
                ("LEADING", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#BBBBBB")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8FAFC")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F4F8FB")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return t


def code_block(text: str, styles) -> Table:
    cleaned = dedent(text).strip()
    t = Table([[Paragraph(cleaned.replace("\n", "<br/>"), styles["SmallCode"])]], colWidths=[7.0 * inch], hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F3F6F9")),
                ("BOX", (0, 0), (-1, -1), 0.35, colors.HexColor("#D0D7DE")),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return t


def build() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleBlue",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=27,
            textColor=colors.HexColor("#17365D"),
            spaceAfter=14,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H1Blue",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#17365D"),
            spaceBefore=14,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H2Blue",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12.5,
            leading=16,
            textColor=colors.HexColor("#2F5597"),
            spaceBefore=10,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyReadable",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.7,
            leading=14,
            alignment=TA_LEFT,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallReadable",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.7,
            leading=12.2,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallCode",
            parent=styles["Code"],
            fontName="Courier",
            fontSize=7.8,
            leading=10,
            textColor=colors.HexColor("#111111"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="Callout",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9.2,
            leading=13,
            textColor=colors.HexColor("#7A3E00"),
            backColor=colors.HexColor("#FFF4E5"),
            borderColor=colors.HexColor("#E69138"),
            borderWidth=0.5,
            borderPadding=7,
            spaceAfter=8,
        )
    )

    story = []
    B = styles["BodyReadable"]
    S = styles["SmallReadable"]

    story.append(p("Beginner Guide to the CH4 Plume Synthetic Publication Project", styles["TitleBlue"]))
    story.append(p("A step-by-step explanation of the synthetic methane plume segmentation benchmark, what each technical term means, why a controlled synthetic approach was chosen, and how the full pipeline works.", B))
    story.append(Spacer(1, 0.15 * inch))
    story.append(FlowDiagram(["Raw Sentinel-2|L1C pairs", "Synthetic plume|injection", "Segmentation|model training", "Validation and|TEST metrics", "Publication|figures"]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(p("Main idea: this project uses real satellite images as backgrounds, injects artificial methane plumes in a controlled way, and trains models to find those injected plumes.", styles["Callout"]))

    story.append(p("1. The Project In One Sentence", styles["H1Blue"]))
    story.append(p("The new project, named CH4_Plume_Synthetic_Publication, is a self-contained benchmark for methane plume segmentation using synthetic plumes injected into real Sentinel-2 L1C event/reference image pairs.", B))
    story.append(p("This means the images are real satellite backgrounds, but the plume labels are not transferred from Carbon Mapper or EMIT. Instead, the project creates its own plume signal and its own perfectly aligned plume mask.", B))

    story.append(p("2. Beginner Definitions", styles["H1Blue"]))
    definitions = [
        ["Term", "Beginner meaning", "Why it matters"],
        ["CH4", "Chemical formula for methane.", "Methane is the gas we are trying to detect as a plume."],
        ["Plume", "A cloud or trail of gas spreading away from a source.", "The model must identify the plume region in the image."],
        ["Segmentation", "Assigning a label to every pixel in an image.", "The output is a map saying plume or not plume for each pixel."],
        ["Synthetic", "Artificially generated, not directly observed.", "The plume is created by code so the label is perfectly known."],
        ["Sentinel-2", "A European Earth-observation satellite mission.", "It provides the optical/SWIR imagery used as background data."],
        ["L1C", "Level-1C top-of-atmosphere Sentinel-2 data.", "It is close to the measured satellite signal before surface correction."],
        ["SWIR", "Short-wave infrared light.", "Methane has absorption behavior in SWIR wavelengths."],
        ["B11/B12", "Sentinel-2 SWIR bands.", "These bands are commonly used in Sentinel-2 methane-sensitive features."],
        ["Event image", "The satellite image near the plume event date.", "This is where the synthetic plume is injected."],
        ["Reference image", "A comparable image without the target event.", "It helps remove background surface patterns."],
        ["GeoTIFF", "A geospatial image file with coordinate information.", "The raw image chips are stored as .tif files."],
        ["Manifest", "A CSV table listing data files and metadata.", "The code uses it to know which files belong to TRAIN, VAL, and TEST."],
        ["TRAIN", "Data used to teach the model.", "The model adjusts its weights using this data."],
        ["VAL", "Validation data used during training.", "It helps choose the best checkpoint."],
        ["TEST", "Held-out data used after training.", "It gives the final benchmark result."],
        ["Tau", "Optical-depth strength of the synthetic methane plume.", "Higher tau means stronger simulated absorption."],
        ["Benchmark", "A fair comparison between models.", "PhysTAUNet is compared against U-Net, Attention U-Net, U-Net++, and DeepLabV3+."],
    ]
    story.append(table(definitions, widths=[1.2 * inch, 2.65 * inch, 3.15 * inch]))

    story.append(PageBreak())
    story.append(p("3. Why A Synthetic Benchmark", styles["H1Blue"]))
    story.append(p("Using real plume labels from Carbon Mapper or EMIT as pixel-level training targets for Sentinel-2 segmentation introduces several alignment problems.", B))
    story.append(bullets([
        "Carbon Mapper and EMIT are different sensors from Sentinel-2 with different pixel sizes and viewing geometry.",
        "A plume label observed by one sensor may shift after reprojection onto the Sentinel-2 grid.",
        "Methane plumes move with wind: a label captured at one time may not match a Sentinel-2 image taken at a different time.",
        "For segmentation tasks, even small spatial shifts degrade precision, recall, and F1 scores.",
        "When two architectures are compared against noisy cross-sensor labels, accuracy differences may reflect label noise rather than architecture quality."
    ], S))
    story.append(p("A controlled synthetic benchmark avoids all of these problems. It creates the plume directly on the Sentinel-2 image grid with a physics-based model, so the label is perfectly aligned, exactly known, and identical for every architecture under test.", B))

    story.append(p("4. Design Choices", styles["H1Blue"]))
    comparison = [
        ["Design choice", "What was chosen", "Why"],
        ["Plume source", "Synthetic (Beer-Lambert physics on the S2 grid).", "Label is pixel-exact; no cross-sensor reprojection error."],
        ["Background imagery", "Real Sentinel-2 L1C event/reference chip pairs.", "Realistic surface texture and atmospheric variability."],
        ["Plume morphology", "Random walks with variable length and dilation width.", "Irregular, non-circular shapes that are non-trivial to segment."],
        ["Optical depth", "Random draw from calibrated ppb-scale range.", "Spans barely detectable to strongly absorbing plumes."],
        ["Train/val/test split", "Spatial blocking (0.25-degree grid, 70/15/15).", "No geographic overlap between splits; prevents leakage."],
        ["Scope", "Architecture comparison on a controlled benchmark.", "Claims are reproducible; no deployment-accuracy claim is made."],
    ]
    story.append(table(comparison, widths=[1.45 * inch, 2.2 * inch, 3.05 * inch]))

    story.append(p("5. What Self-Contained Means Here", styles["H1Blue"]))
    story.append(p("Self-contained means the project acquires all data from public sources and builds everything from scratch. Raw Sentinel-2 chips are downloaded from Google Earth Engine (a free public satellite archive). Training manifests are generated by this project's own scripts. No other local project is required.", B))
    story.append(code_block("""
    CH4_Plume_Synthetic_Publication/
      data/raw/sentinel2_l1c/exports/      downloaded S2 L1C GeoTIFF chips
      data/raw/splits/                     split catalog CSV (plume event list)
      data/training_l1c/                   aligned masks and curated manifest
      data/models/                         trained model checkpoints
      data/outputs/                        tables and publication figures
      reports/                             markdown and PDF reports
      src/ch4l1c/                          data acquisition + training code
      scripts/acquire_s2_data.sh           Step 1: download chips from GEE
      scripts/build_training_manifest.sh   Step 2: align masks + curate
      scripts/run_all.sh                   Steps 2-3 combined entrypoint
    """, styles))

    story.append(PageBreak())
    story.append(p("6. Step-By-Step Pipeline", styles["H1Blue"]))
    story.append(FlowDiagram(["Query GEE|for S2 chips", "Download|via rclone", "Align masks|to chip grids", "Inject synthetic|methane plumes", "Train|models", "Evaluate|VAL and TEST", "Create|figures"], width=7.0 * inch, height=1.15 * inch))
    steps = [
        ("Step 1a - Query Google Earth Engine", "The script ch4l1c s2-l1c-reference-match finds the best low-cloud Sentinel-2 L1C event and reference scene for each plume event in the split catalog."),
        ("Step 1b - Export chips to Google Drive", "The script ch4l1c s2-l1c-export-pairs queues 512x512 pixel chip exports (12 bands: event + reference B2-B12) to a Google Drive folder via GEE tasks."),
        ("Step 1c - Download chips", "Once GEE tasks complete, rclone copies the GeoTIFF chips from Google Drive to data/raw/sentinel2_l1c/exports/."),
        ("Step 2 - Align plume masks", "The script ch4l1c build-training-dataset reprojects Carbon Mapper / EMIT plume masks onto each chip's pixel grid and writes aligned mask files."),
        ("Step 3 - Curate manifest", "The script ch4l1c curate-training-manifest adds quality flags, sample weights, and recommended roles (TRAIN/VAL/TEST) to the manifest CSV."),
        ("Step 4 - Generate synthetic plume", "For every training patch, the code creates an artificial plume field with irregular shape."),
        ("Step 5 - Inject methane absorption", "The code modifies the methane-sensitive SWIR signal to simulate the effect of methane absorption."),
        ("Step 6 - Train models", "PhysTAUNet and benchmark models learn to predict the synthetic plume mask."),
        ("Step 7 - Validate and test", "Validation tracks training quality. TEST gives held-out final benchmark metrics."),
        ("Step 8 - Create outputs", "The pipeline writes CSV tables, reports, metric plots, and qualitative example figures."),
    ]
    for title, body in steps:
        story.append(KeepTogether([p(title, styles["H2Blue"]), p(body, B)]))

    story.append(p("7. The Synthetic Plume Method", styles["H1Blue"]))
    story.append(p("The method is physics-inspired. It does not pretend that the synthetic plume is an observed real plume. Instead, it uses a simplified methane absorption model to create a controlled learning task.", B))
    story.append(p("The most important formula in the current code is the attenuation step:", B))
    story.append(code_block("""
    absorption = exp(-tau * plume_field)
    injected_B12 = event_B12 * absorption
    """, styles))
    story.append(p("Beginner meaning: the synthetic plume reduces the methane-sensitive B12 signal. The reduction is stronger where the plume field is stronger and where tau is larger.", B))
    story.append(bullets([
        "plume_field is the spatial shape of the plume, with higher values near stronger plume regions.",
        "tau controls the strength of methane absorption.",
        "exp means exponential function. It makes absorption smoothly decrease as tau and plume strength increase.",
        "event_B12 is the original Sentinel-2 B12 band in the event image.",
        "injected_B12 is the modified B12 band after the synthetic methane plume is added."
    ], S))

    story.append(p("8. Why The Shape Generator Was Changed", styles["H1Blue"]))
    story.append(p("Earlier synthetic examples looked too regular, almost like smooth ellipses. That is not realistic enough because real methane plumes are shaped by wind, turbulence, source behavior, and background noise.", B))
    story.append(p("The current generator creates more realistic variation:", B))
    story.append(bullets([
        "meandering centerline - the plume path curves instead of staying straight",
        "source/core region - a stronger area close to the emission source",
        "width variation - the plume can widen or narrow downwind",
        "detached wisps - small separated patches can appear away from the main plume",
        "turbulence texture - the intensity is not perfectly smooth",
        "holes and ragged edges - boundaries are irregular instead of clean ellipses",
        "random thresholding - each mask can have a different final outline"
    ], S))

    story.append(PageBreak())
    story.append(p("9. What The Model Actually Learns", styles["H1Blue"]))
    story.append(p("The model is not learning from real Carbon Mapper plume masks in this project. It learns a controlled task: given a real Sentinel-2 background with synthetic methane absorption injected, predict the synthetic plume mask.", B))
    story.append(code_block("""
    Input to model:
      real Sentinel-2 L1C event/reference features
      plus synthetic methane absorption signal

    Target label:
      synthetic plume mask created from the same plume field

    Model output:
      probability map, where each pixel receives a plume probability
    """, styles))
    story.append(p("A probability map is not immediately a final mask. A threshold converts probabilities into plume/non-plume pixels. For example, if threshold = 0.5, then pixels with probability >= 0.5 are predicted as plume.", B))

    story.append(p("10. Models In The Benchmark", styles["H1Blue"]))
    models = [
        ["Model", "Beginner explanation"],
        ["U-Net", "A common medical and remote-sensing segmentation model. It uses an encoder-decoder shape with skip connections."],
        ["Attention U-Net", "A U-Net variant that adds attention gates, helping the model focus on relevant regions."],
        ["U-Net++", "A U-Net variant with more nested skip pathways, designed to improve feature reuse."],
        ["DeepLabV3+", "A segmentation model that uses atrous/dilated convolutions to capture broader context."],
        ["PhysTAUNet", "The custom model. It is meant to use methane-sensitive physics features and segmentation learning together."],
    ]
    story.append(table(models, widths=[1.45 * inch, 5.55 * inch]))

    story.append(p("11. Metrics Explained", styles["H1Blue"]))
    metrics = [
        ["Metric", "Meaning", "Beginner interpretation"],
        ["Precision", "Of all pixels predicted as plume, how many were truly plume?", "High precision means few false plume pixels."],
        ["Recall", "Of all true plume pixels, how many did the model find?", "High recall means the model misses fewer plume pixels."],
        ["F1", "A balance between precision and recall.", "Useful when plume pixels are rare."],
        ["IoU", "Intersection over Union between predicted and true plume masks.", "A stricter spatial-overlap score."],
        ["Tolerant F1", "F1 after allowing small spatial mismatch.", "Useful because plume boundaries can be uncertain."],
        ["Predicted positive fraction", "Fraction of pixels predicted as plume.", "If this is near 1.0, the model is predicting almost everything as plume."],
    ]
    story.append(table(metrics, widths=[1.35 * inch, 2.6 * inch, 3.05 * inch]))

    story.append(p("12. Why The Smoke Test Looked Bad", styles["H1Blue"]))
    story.append(p("The smoke test used only 20 raw files, 1 epoch, and 64 patches per epoch. That run was only meant to check whether the pipeline works. It was not meant to produce a scientific result.", B))
    story.append(p("In that smoke run, predicted_positive_fraction was close to 1.0. That means the model learned a trivial behavior: predict nearly every pixel as plume. This is a common early-training problem when the run is too short.", B))
    story.append(p("The real benchmark needs many more patches and more epochs, for example 40 epochs and 8192 patches per epoch.", B))

    story.append(PageBreak())
    story.append(p("13. Commands To Reproduce", styles["H1Blue"]))
    story.append(p("Install and activate environment:", styles["H2Blue"]))
    story.append(code_block("""
    cd ~/CH4_Plume_Synthetic_Publication
    source /home/ubuntu/.venv/bin/activate
    python -m pip install -r requirements.txt
    python -m pip install -e .
    """, styles))
    story.append(p("Full pipeline from scratch (acquire data then train):", styles["H2Blue"]))
    story.append(code_block("""
    cd ~/CH4_Plume_Synthetic_Publication
    source /home/ubuntu/.venv/bin/activate

    # Step 1: download Sentinel-2 chips from Google Earth Engine
    GEE_PROJECT=your-gcp-project \\
    DRIVE_FOLDER=CH4_Plume_L1C_S2_pairs \\
    bash scripts/acquire_s2_data.sh

    # Steps 2-3: build manifest and train
    EPOCHS=40 \\
    BATCH_SIZE=8 \\
    PATCH_SIZE=128 \\
    PATCHES_PER_EPOCH=8192 \\
    VALIDATION_PATCHES=1024 \\
    TEST_PATCHES=2048 \\
    MODELS="phys_tau_net unet attn_unet unet_pp deeplabv3p" \\
    bash scripts/run_all.sh
    """, styles))
    story.append(p("Check outputs after training:", styles["H2Blue"]))
    story.append(code_block("""
    cat data/outputs/tables/synthetic_publication_benchmark_summary.csv
    ls -lh data/outputs/publication_figures/
    ls -lh data/outputs/publication_figures/synthetic_segmentation_examples/
    """, styles))

    story.append(p("14. Outputs Explained", styles["H1Blue"]))
    outputs = [
        ["Output", "What it contains"],
        ["data/models/synthetic_publication_benchmark/{model}/best.pt", "Best saved model checkpoint for that architecture."],
        ["data/models/synthetic_publication_benchmark/{model}/history.csv", "Validation metrics after each epoch."],
        ["data/models/synthetic_publication_benchmark/{model}/synthetic_eval_test.csv", "Held-out synthetic TEST metrics."],
        ["data/outputs/tables/synthetic_publication_benchmark_summary.csv", "Combined benchmark table across all models."],
        ["data/outputs/publication_figures/metrics_synthetic_segmentation.png", "Validation metric figure."],
        ["data/outputs/publication_figures/metrics_synthetic_test_segmentation.png", "Held-out TEST metric figure."],
        ["data/outputs/publication_figures/synthetic_segmentation_examples/", "Qualitative segmentation examples for publication review."],
        ["reports/synthetic_publication_benchmark_report.md", "Markdown report summarizing the benchmark."],
    ]
    story.append(table(outputs, widths=[3.2 * inch, 3.8 * inch]))

    story.append(PageBreak())
    story.append(p("15. What Can Be Claimed In A Paper", styles["H1Blue"]))
    story.append(p("A safe and accurate publication claim is:", styles["H2Blue"]))
    story.append(p("PhysTAUNet was benchmarked against U-Net, Attention U-Net, U-Net++, and DeepLabV3+ on a controlled synthetic Sentinel-2 L1C methane plume segmentation task. Synthetic plumes were injected into real Sentinel-2 L1C event/reference backgrounds using methane-sensitive SWIR attenuation and irregular plume morphology.", styles["Callout"]))
    story.append(p("Do not claim that this project proves operational real-world plume segmentation from Carbon Mapper or EMIT labels. That would require a separately validated real-label dataset.", B))

    story.append(p("16. Limitations", styles["H1Blue"]))
    story.append(bullets([
        "Synthetic plumes are controlled and useful for benchmarking, but they are not the same as observed real plumes.",
        "The injection model simplifies methane radiative transfer. It is physics-inspired, not a full atmospheric radiative-transfer simulator.",
        "Model performance on synthetic TEST data does not automatically prove performance on real methane events.",
        "Future real-world validation still needs carefully matched, pixel-reliable plume labels.",
        "The paper should be clear that the main claim is synthetic Sentinel-2 plume segmentation."
    ], S))

    story.append(p("17. Recommended Next Work", styles["H1Blue"]))
    story.append(bullets([
        "Improve the synthetic generator using wind direction, source location, and physically parameterized plume spread.",
        "Compare synthetic plume shapes against real Carbon Mapper and EMIT morphology statistics.",
        "Add ablation studies: with/without event-reference features, with/without tau physics features, and with different plume strengths.",
        "Create a small manually reviewed real Sentinel-2 validation set if possible.",
        "Use the synthetic benchmark to justify PhysTAUNet architecture, then treat real-data testing as a separate follow-up study."
    ], S))

    story.append(p("18. References For Method Framing", styles["H1Blue"]))
    story.append(bullets([
        "Climate Change AI / NeurIPS 2023: Methane plume detection with U-Net segmentation on Sentinel-2. https://www.climatechange.ai/papers/neurips2023/78",
        "Ruzicka et al. 2023: Semantic segmentation of methane plumes with hyperspectral machine learning models. https://www.nature.com/articles/s41598-023-44918-6",
        "Wang et al. 2024: Matched filter for Sentinel-2 methane plume detection. https://www.mdpi.com/2072-4292/16/6/1023",
        "Sentinel-2 methane monitoring physics and B11/B12 transmittance context. https://amt.copernicus.org/articles/16/89/2023/"
    ], S))

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=letter,
        rightMargin=0.62 * inch,
        leftMargin=0.62 * inch,
        topMargin=0.62 * inch,
        bottomMargin=0.68 * inch,
        title="Beginner Guide to CH4 Plume Synthetic Publication",
        author="Codex",
    )
    doc.build(story, onFirstPage=add_header_footer, onLaterPages=add_header_footer)


if __name__ == "__main__":
    build()
    print(OUT)
