import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { Canvas } = require("../node_modules/@oai/artifact-tool/node_modules/skia-canvas");

const {
  Presentation,
  PresentationFile,
  row,
  column,
  grid,
  layers,
  panel: rawPanel,
  text,
  image: rawImage,
  shape: rawShape,
  rule,
  fill,
  hug,
  fixed,
  wrap,
  grow,
  fr,
  drawSlideToCtx,
} = await import("@oai/artifact-tool");
const { paint, stroke } = await import("@oai/artifact-tool/presentation-jsx");

const WORKSPACE = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const REPO = path.resolve(WORKSPACE, "..", "..");
const OUTPUT_DIR = path.join(WORKSPACE, "output");
const FINAL_NAME = "SC26_CDC_Yifan_Tiling_Progress_2026-05-12.pptx";
const OUTPUT = path.join(OUTPUT_DIR, FINAL_NAME);
const ROOT_COPY = path.join(WORKSPACE, FINAL_NAME);
const PREVIEW_DIR = path.join(WORKSPACE, "scratch", "previews");

const COLORS = {
  bg: "#F7F9FC",
  ink: "#172033",
  muted: "#5B667A",
  faint: "#E4E9F2",
  white: "#FFFFFF",
  navy: "#0B1B32",
  blue: "#2563EB",
  teal: "#0F766E",
  green: "#15803D",
  orange: "#D97706",
  red: "#B91C1C",
  lavender: "#EEF2FF",
  tealSoft: "#E7F6F2",
  orangeSoft: "#FFF7ED",
  blueSoft: "#EAF1FF",
  graySoft: "#F1F5F9",
};

const SLIDE = { width: 1920, height: 1080 };
const BODY_FONT = "Aptos";
const DISPLAY_FONT = "Aptos Display";

const rows = [
  {
    setup: "No tiling",
    tile: "None",
    time: 143.55,
    memory: 52.0,
    ratio: 72.74,
    psnr: 29.88,
    ssim: 0.8847,
    seam: "n/a",
  },
  {
    setup: "512 tile",
    tile: "512",
    time: 86.01,
    memory: 3.0,
    ratio: 68.79,
    psnr: 29.73,
    ssim: 0.8822,
    seam: "0.027796",
  },
  {
    setup: "1024 tile",
    tile: "1024",
    time: 88.35,
    memory: 11.2,
    ratio: 66.11,
    psnr: 29.82,
    ssim: 0.8835,
    seam: "0.028595",
  },
  {
    setup: "2048 tile",
    tile: "2048",
    time: 95.39,
    memory: 43.8,
    ratio: 66.04,
    psnr: 29.9,
    ssim: 0.8841,
    seam: "0.031026",
  },
];

function asFill(value) {
  return typeof value === "string" ? paint(value) : value;
}

function asLine(value) {
  return typeof value === "string" ? stroke(value) : value;
}

function panel(options = {}, child) {
  return rawPanel(
    {
      ...options,
      fill: asFill(options.fill),
      line: asLine(options.line),
    },
    child,
  );
}

function shape(options = {}) {
  return rawShape({
    ...options,
    fill: asFill(options.fill),
    line: asLine(options.line),
  });
}

function image(options = {}) {
  if (options.path) {
    const ext = path.extname(options.path).toLowerCase();
    const mime = ext === ".jpg" || ext === ".jpeg" ? "image/jpeg" : "image/png";
    const dataUrl = `data:${mime};base64,${fs.readFileSync(options.path).toString("base64")}`;
    const { path: _path, ...rest } = options;
    return rawImage({ ...rest, dataUrl, contentType: mime });
  }
  return rawImage(options);
}

function pathTo(relativePath) {
  return path.join(REPO, relativePath);
}

function t(value, options = {}) {
  return text(value, {
    width: options.width ?? fill,
    height: options.height ?? hug,
    name: options.name,
    style: {
      fontFamily: options.fontFamily ?? BODY_FONT,
      fontSize: options.size ?? 28,
      color: options.color ?? COLORS.ink,
      bold: options.bold ?? false,
      italic: options.italic ?? false,
    },
    columnSpan: options.columnSpan,
    rowSpan: options.rowSpan,
  });
}

function badge(label, fillColor = COLORS.blueSoft, color = COLORS.blue) {
  return panel(
    {
      width: hug,
      height: hug,
      padding: { x: 18, y: 8 },
      fill: fillColor,
      borderRadius: 8,
      line: fillColor,
    },
    t(label, { width: hug, size: 19, bold: true, color }),
  );
}

function titleBlock(title, subtitle) {
  return column({ width: fill, height: hug, gap: 12 }, [
    t(title, {
      name: "slide-title",
      size: 50,
      bold: true,
      fontFamily: DISPLAY_FONT,
      color: COLORS.ink,
    }),
    subtitle
      ? t(subtitle, {
          name: "slide-subtitle",
          size: 24,
          color: COLORS.muted,
          width: wrap(1320),
        })
      : null,
  ].filter(Boolean));
}

function footer(source) {
  return row({ width: fill, height: hug, align: "center", justify: "between" }, [
    t(source, { size: 15, color: "#6B7280", width: wrap(1300) }),
    t("CDC tiling weekly progress | May 12, 2026", {
      size: 15,
      color: "#6B7280",
      width: hug,
    }),
  ]);
}

function bullet(label, detail, color = COLORS.blue) {
  return row({ width: fill, height: hug, gap: 14, align: "start" }, [
    shape({ width: fixed(12), height: fixed(12), fill: color, line: color, borderRadius: 6 }),
    column({ width: fill, height: hug, gap: 4 }, [
      t(label, { size: 23, bold: true, color: COLORS.ink }),
      detail ? t(detail, { size: 19, color: COLORS.muted }) : null,
    ].filter(Boolean)),
  ]);
}

function metricCard(label, value, detail, accent = COLORS.blue, fillColor = COLORS.white) {
  return panel(
    {
      width: fill,
      height: fill,
      fill: fillColor,
      line: COLORS.faint,
      borderRadius: 8,
      padding: { x: 22, y: 18 },
    },
    column({ width: fill, height: fill, gap: 8, justify: "between" }, [
      t(label, { size: 18, bold: true, color: COLORS.muted }),
      t(value, {
        size: 34,
        bold: true,
        color: accent,
        fontFamily: DISPLAY_FONT,
      }),
      t(detail, { size: 18, color: COLORS.muted }),
    ]),
  );
}

function evidencePanel(children, fillColor = COLORS.white) {
  return panel(
    {
      width: fill,
      height: fill,
      fill: fillColor,
      line: COLORS.faint,
      borderRadius: 8,
      padding: { x: 26, y: 24 },
    },
    column({ width: fill, height: fill, gap: 18 }, children),
  );
}

function slideShell(presentation, title, subtitle, body, source) {
  const slide = presentation.slides.add();
  slide.compose(
    layers({ name: "slide-bg", width: fill, height: fill }, [
      shape({ name: "background", width: fill, height: fill, fill: COLORS.bg, line: COLORS.bg }),
      column(
        {
          name: "content-root",
          width: fill,
          height: fill,
          padding: { x: 72, y: 54 },
          gap: 28,
        },
        [
          titleBlock(title, subtitle),
          rule({ width: fixed(180), stroke: COLORS.blue, weight: 4 }),
          ...body,
          footer(source),
        ],
      ),
    ]),
    { frame: { left: 0, top: 0, width: SLIDE.width, height: SLIDE.height }, baseUnit: 8 },
  );
  return slide;
}

function fmt(value, digits = 1) {
  return Number(value).toFixed(digits);
}

function pct(value, digits = 0) {
  return `${Number(value).toFixed(digits)}%`;
}

function reduction(baseline, candidate) {
  return ((baseline - candidate) / baseline) * 100;
}

function tableRows(headers, rowsData, widths, options = {}) {
  const cols = widths.map((width) => fr(width));
  const makeCell = (cell, fillColor, size, bold = false, color = COLORS.ink) =>
    panel(
      {
        width: fill,
        height: fill,
        fill: fillColor,
        line: COLORS.faint,
        padding: { x: 12, y: 8 },
      },
      t(String(cell), { size, bold, color, width: fill }),
    );

  return column({ width: fill, height: hug, gap: 2 }, [
    grid(
      { width: fill, height: fixed(options.headerHeight ?? 42), columns: cols },
      headers.map((cell) => makeCell(cell, COLORS.navy, options.headerSize ?? 16, true, COLORS.white)),
    ),
    ...rowsData.map((cells, rowIndex) =>
      grid(
        { width: fill, height: fixed(options.rowHeight ?? 48), columns: cols },
        cells.map((cell) => makeCell(cell, rowIndex % 2 === 0 ? COLORS.white : COLORS.graySoft, options.bodySize ?? 17)),
      ),
    ),
  ]);
}

function barRow(label, value, maxValue, color, suffix, detail) {
  const width = Math.max(16, Math.round((value / maxValue) * 520));
  return column({ width: fill, height: hug, gap: 6 }, [
    row({ width: fill, height: hug, justify: "between", align: "center" }, [
      t(label, { size: 20, bold: true, width: hug }),
      t(`${fmt(value, value >= 100 ? 0 : 1)}${suffix}`, { size: 20, bold: true, color, width: hug }),
    ]),
    row({ width: fill, height: fixed(24), align: "center" }, [
      shape({ width: fixed(width), height: fixed(18), fill: color, line: color, borderRadius: 5 }),
      shape({ width: fill, height: fixed(2), fill: COLORS.faint, line: COLORS.faint }),
    ]),
    detail ? t(detail, { size: 15, color: COLORS.muted }) : null,
  ].filter(Boolean));
}

function addCover(presentation) {
  const slide = presentation.slides.add();
  slide.compose(
    layers({ width: fill, height: fill }, [
      shape({ width: fill, height: fill, fill: COLORS.navy, line: COLORS.navy }),
      grid(
        {
          width: fill,
          height: fill,
          columns: [fr(1.05), fr(0.95)],
          columnGap: 54,
          padding: { x: 76, y: 70 },
        },
        [
          column({ width: fill, height: fill, gap: 28, justify: "center" }, [
            row({ width: fill, height: hug, gap: 12 }, [
              badge("SC26 CDC", "#12345C", "#BFE8FF"),
              badge("Yifan tiling", "#163B32", "#C9F4E8"),
            ]),
            t("Tiling Makes Full-Resolution Compression Practical", {
              width: wrap(840),
              size: 70,
              bold: true,
              color: COLORS.white,
              fontFamily: DISPLAY_FONT,
            }),
            t("DeltaAI GH200 pilot shows 512 x 512 tiles cut runtime and memory without obvious stitching artifacts in the checked sample.", {
              width: wrap(820),
              size: 29,
              color: "#D7E3F5",
            }),
            grid({ width: fill, height: fixed(178), columns: [fr(1), fr(1), fr(1)], columnGap: 18 }, [
              metricCard("Runtime", "86.01s", "per image with 512 tiles", "#7DD3FC", "#102848"),
              metricCard("Memory", "3.0 GB", "17.2x lower than no tiling", "#86EFAC", "#102848"),
              metricCard("Artifacts", "No obvious seams", "checked visual sample", "#FDBA74", "#102848"),
            ]),
            t("Prepared for the week ending May 12, 2026", {
              size: 21,
              color: "#B9C7DD",
            }),
          ]),
          panel(
            {
              width: fill,
              height: fill,
              fill: COLORS.white,
              line: "#223B5C",
              borderRadius: 8,
              padding: { x: 18, y: 18 },
            },
            column({ width: fill, height: fill, gap: 16 }, [
              image({
                path: pathTo("results/2026-05-12-yifan-tiling-pilot/visual_examples_small/100_0005_0001_tile512_overview.jpg"),
                width: fill,
                height: grow(1),
                fit: "cover",
                alt: "512 tile stitched overview",
              }),
              t("Visual evidence: stitched 512 x 512 tiled output, image 100_0005_0001", {
                size: 18,
                color: COLORS.muted,
              }),
            ]),
          ),
        ],
      ),
    ]),
    { frame: { left: 0, top: 0, width: SLIDE.width, height: SLIDE.height }, baseUnit: 8 },
  );
}

function addWorkflowSlide(presentation) {
  slideShell(
    presentation,
    "This Week's Task Was Narrow and Testable",
    "We focused only on Yifan's tiling path: split, compress independently, stitch, then check speed, memory, quality, and seams.",
    [
      grid({ width: fill, height: grow(1), columns: [fr(1), fr(1), fr(1), fr(1)], columnGap: 18 }, [
        evidencePanel([
          badge("1", COLORS.blueSoft, COLORS.blue),
          t("Split", { size: 31, bold: true }),
          t("Full-resolution images are cropped to 5440 x 3648, then split into 512, 1024, or 2048 patches.", { size: 21, color: COLORS.muted }),
        ]),
        evidencePanel([
          badge("2", COLORS.tealSoft, COLORS.teal),
          t("Compress", { size: 31, bold: true }),
          t("Each tile runs through the pretrained baseline_b02048 checkpoint with 65 denoising steps.", { size: 21, color: COLORS.muted }),
        ]),
        evidencePanel([
          badge("3", COLORS.orangeSoft, COLORS.orange),
          t("Stitch", { size: 31, bold: true }),
          t("Tiles are assembled back into one image and saved with visual examples for inspection.", { size: 21, color: COLORS.muted }),
        ]),
        evidencePanel([
          badge("4", COLORS.lavender, COLORS.blue),
          t("Measure", { size: 31, bold: true }),
          t("Summary records runtime, peak GPU memory, compression ratio, PSNR, SSIM, and seam metrics.", { size: 21, color: COLORS.muted }),
        ]),
      ]),
      grid({ width: fill, height: fixed(150), columns: [fr(1), fr(1), fr(1), fr(1)], columnGap: 16 }, [
        metricCard("Pilot size", "8 images", "after one-image smoke test", COLORS.blue),
        metricCard("GPU target", "GH200", "DeltaAI ghx4 partition", COLORS.teal),
        metricCard("Reference", "No tiling", "full-image baseline", COLORS.orange),
        metricCard("Tile sizes", "512 / 1024 / 2048", "all completed", COLORS.green),
      ]),
    ],
    "Source: experiments/compression/slurm/03_tiling_sweep.sbatch and run stamp 20260512_yifan_tiling_pilot.",
  );
}

function addPilotResultsSlide(presentation) {
  const baseline = rows[0];
  const tile512 = rows[1];
  slideShell(
    presentation,
    "512 x 512 Is the Current Speed and Memory Winner",
    "Across the eight-image pilot, smaller tiles gave the fastest runtime and the largest memory reduction.",
    [
      grid({ width: fill, height: grow(1), columns: [fr(0.95), fr(1.05)], columnGap: 28 }, [
        evidencePanel([
          t("Pilot Summary", { size: 30, bold: true }),
          tableRows(
            ["Setup", "Time", "Memory", "Ratio", "PSNR", "SSIM"],
            rows.map((item) => [
              item.setup,
              `${fmt(item.time, 2)}s`,
              `${fmt(item.memory, 1)} GB`,
              `${fmt(item.ratio, 2)}x`,
              fmt(item.psnr, 2),
              fmt(item.ssim, 4),
            ]),
            [1.15, 0.72, 0.82, 0.72, 0.62, 0.66],
            { rowHeight: 52, bodySize: 17 },
          ),
        ]),
        evidencePanel([
          t("What the Numbers Say", { size: 30, bold: true }),
          barRow("No tiling runtime", baseline.time, baseline.time, COLORS.muted, "s", "Full-image reference"),
          barRow("512 tile runtime", tile512.time, baseline.time, COLORS.teal, "s", `${pct(reduction(baseline.time, tile512.time), 1)} lower wall time`),
          barRow("No tiling memory", baseline.memory, baseline.memory, COLORS.muted, " GB", "Full-image reference"),
          barRow("512 tile memory", tile512.memory, baseline.memory, COLORS.green, " GB", "17.2x lower peak GPU memory"),
          t("Trade-off: compression ratio decreases from 72.74x to 68.79x, while PSNR and SSIM remain close to no tiling.", {
            size: 21,
            color: COLORS.muted,
          }),
        ]),
      ]),
    ],
    "Source: results/2026-05-12-yifan-tiling-pilot/tables/combined_summary.csv.",
  );
}

function addVisualCheckSlide(presentation) {
  slideShell(
    presentation,
    "Visual Check Did Not Show New Grid-Like Seams",
    "The seam-region crop compares the same road/intersection area across no tiling and the recommended 512-tile output.",
    [
      grid({ width: fill, height: fixed(560), columns: [fr(1), fr(1)], columnGap: 26 }, [
        panel(
          {
            width: fill,
            height: fill,
            fill: COLORS.white,
            line: COLORS.faint,
            borderRadius: 8,
            padding: { x: 18, y: 16 },
          },
          column({ width: fill, height: fill, gap: 12 }, [
            t("No tiling reference", { size: 24, bold: true }),
            image({
              path: pathTo("results/2026-05-12-yifan-tiling-pilot/visual_examples_small/100_0005_0001_no_tiling_seam_region.jpg"),
              width: fill,
              height: fixed(438),
              fit: "contain",
              alt: "No tiling seam-region crop",
            }),
          ]),
        ),
        panel(
          {
            width: fill,
            height: fill,
            fill: COLORS.white,
            line: COLORS.faint,
            borderRadius: 8,
            padding: { x: 18, y: 16 },
          },
          column({ width: fill, height: fill, gap: 12 }, [
            t("512 tile stitched output", { size: 24, bold: true }),
            image({
              path: pathTo("results/2026-05-12-yifan-tiling-pilot/visual_examples_small/100_0005_0001_tile512_seam_region.jpg"),
              width: fill,
              height: fixed(438),
              fit: "contain",
              alt: "512 tile seam-region crop",
            }),
          ]),
        ),
      ]),
      grid({ width: fill, height: fixed(150), columns: [fr(1), fr(1), fr(1)], columnGap: 16 }, [
        metricCard("Seam metric", "0.0278", "512-tile average", COLORS.teal),
        metricCard("Visual finding", "Pass", "no obvious grid pattern", COLORS.green),
        metricCard("Remaining caution", "Sample check", "larger visual review still useful", COLORS.orange),
      ]),
    ],
    "Source: visual examples in results/2026-05-12-yifan-tiling-pilot/visual_examples_small/.",
  );
}

function addDecisionSlide(presentation) {
  slideShell(
    presentation,
    "Decision: Use 512 x 512 as the Default Tiling Setup",
    "The current evidence favors 512 x 512 for speed and memory, with 1024 x 1024 as a backup if larger visual checks reveal artifacts.",
    [
      grid({ width: fill, height: grow(1), columns: [fr(1), fr(1)], columnGap: 28 }, [
        evidencePanel([
          badge("Recommended", COLORS.tealSoft, COLORS.teal),
          bullet("Default setup: 512 x 512 tiles.", "Fastest runtime and largest memory drop in the eight-image pilot.", COLORS.teal),
          bullet("Use this result in the weekly update.", "86.01s/image, 3.0 GB, compression ratio 68.79x, PSNR 29.73, SSIM 0.8822.", COLORS.blue),
          bullet("Mention artifact check honestly.", "Checked examples show no obvious grid-like seams; larger-sample review remains a final-poster step.", COLORS.green),
        ]),
        evidencePanel([
          badge("Next", COLORS.orangeSoft, COLORS.orange),
          bullet("Run larger selected setup.", "Suggested run: N_IMAGES=50, primarily 512 x 512.", COLORS.orange),
          bullet("Keep 1024 x 1024 as backup.", "It is slightly slower and uses more memory, but quality metrics are close to no tiling.", COLORS.orange),
          bullet("Prepare poster table.", "Report time, memory, compression ratio, PSNR, SSIM, and artifact status.", COLORS.orange),
        ]),
      ]),
      grid({ width: fill, height: fixed(140), columns: [fr(1), fr(1), fr(1)], columnGap: 16 }, [
        metricCard("Runtime reduction", "40.1%", "143.55s to 86.01s", COLORS.teal),
        metricCard("Memory reduction", "17.2x", "52.0 GB to 3.0 GB", COLORS.green),
        metricCard("Current answer", "Tiling helps", "fast and storage-aware path", COLORS.blue),
      ]),
    ],
    "Source: committed pilot tables, visual examples, and weekly progress note.",
  );
}

function buildDeck() {
  const presentation = Presentation.create({
    slideSize: { width: SLIDE.width, height: SLIDE.height },
  });

  addCover(presentation);
  addWorkflowSlide(presentation);
  addPilotResultsSlide(presentation);
  addVisualCheckSlide(presentation);
  addDecisionSlide(presentation);

  return presentation;
}

async function renderPreviews(presentation) {
  fs.mkdirSync(PREVIEW_DIR, { recursive: true });
  const slides = presentation.slides.items ?? presentation.slides;
  for (let index = 0; index < slides.length; index += 1) {
    const canvas = new Canvas(SLIDE.width, SLIDE.height);
    await drawSlideToCtx(slides[index], presentation, canvas.getContext("2d"));
    await canvas.toFile(path.join(PREVIEW_DIR, `slide_${String(index + 1).padStart(2, "0")}.png`));
  }
}

async function main() {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  fs.mkdirSync(PREVIEW_DIR, { recursive: true });

  const presentation = buildDeck();
  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(OUTPUT);
  fs.copyFileSync(OUTPUT, ROOT_COPY);
  await renderPreviews(presentation);

  console.log(`Saved PPTX: ${OUTPUT}`);
  console.log(`Saved PPTX copy: ${ROOT_COPY}`);
  console.log(`Saved previews: ${PREVIEW_DIR}`);
}

await main();
