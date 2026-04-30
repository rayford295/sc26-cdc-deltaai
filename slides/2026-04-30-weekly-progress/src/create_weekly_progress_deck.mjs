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
  chart,
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
const OUTPUT = path.join(WORKSPACE, "output", "output.pptx");
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

function csvRows(relativePath) {
  const file = path.join(REPO, relativePath);
  const lines = fs.readFileSync(file, "utf8").trim().split(/\r?\n/);
  const headers = lines.shift().split(",");
  return lines.map((line) => {
    const values = line.split(",");
    const out = {};
    headers.forEach((header, index) => {
      const raw = values[index];
      if (raw === "inf") out[header] = Number.POSITIVE_INFINITY;
      else if (/^-?\d+(?:\.\d+)?$/.test(raw)) out[header] = Number(raw);
      else out[header] = raw;
    });
    return out;
  });
}

const gh200 = csvRows("results/2026-04-26-reconstruction/tables/sweep_summary.csv");
const h200 = csvRows("results/2026-04-28-h200-reconstruction/tables/sweep_summary.csv");
const h200Batch = csvRows("results/2026-04-28-h200-reconstruction/tables/batch_pilot_summary.csv");

function metric(rows, precision, steps, key) {
  return rows.find((rowItem) => rowItem.precision === precision && rowItem.n_denoise_step === steps)[key];
}

function fmt(value, digits = 1) {
  return Number(value).toFixed(digits);
}

function gb(mb) {
  return `${fmt(mb / 1000, 1)} GB`;
}

function speedup(baseline, candidate) {
  return ((baseline - candidate) / baseline) * 100;
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
    t("CDC reconstruction weekly progress | Apr 30, 2026", {
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
      t(label, { size: 24, bold: true, color: COLORS.ink }),
      detail ? t(detail, { size: 20, color: COLORS.muted }) : null,
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
        size: 32,
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

function tableRows(headers, rowsData, widths, options = {}) {
  const cols = widths.map((width) => fr(width));
  const header = row(
    { width: fill, height: fixed(options.headerHeight ?? 42), gap: 0 },
    headers.map((cell, index) =>
      panel(
        {
          width: fill,
          height: fill,
          fill: options.headerFill ?? COLORS.navy,
          line: options.headerFill ?? COLORS.navy,
          padding: { x: 14, y: 8 },
        },
        t(cell, {
          size: options.headerSize ?? 17,
          bold: true,
          color: COLORS.white,
          width: fill,
        }),
      ),
    ),
  );

  return column({ width: fill, height: hug, gap: 2 }, [
    grid({ width: fill, height: fixed(options.headerHeight ?? 42), columns: cols }, header.children),
    ...rowsData.map((cells, rowIndex) =>
      grid(
        {
          width: fill,
          height: fixed(options.rowHeight ?? 48),
          columns: cols,
        },
        cells.map((cell) =>
          panel(
            {
              width: fill,
              height: fill,
              fill: rowIndex % 2 === 0 ? COLORS.white : COLORS.graySoft,
              line: COLORS.faint,
              padding: { x: 14, y: 10 },
            },
            t(String(cell), {
              size: options.bodySize ?? 18,
              color: COLORS.ink,
              width: fill,
            }),
          ),
        ),
      ),
    ),
  ]);
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

function addCover(presentation) {
  const slide = presentation.slides.add();
  slide.compose(
    layers({ width: fill, height: fill }, [
      shape({ width: fill, height: fill, fill: COLORS.navy, line: COLORS.navy }),
      grid(
        {
          width: fill,
          height: fill,
          columns: [fr(1.08), fr(0.92)],
          columnGap: 54,
          padding: { x: 76, y: 70 },
        },
        [
          column({ width: fill, height: fill, gap: 28, justify: "center" }, [
            row({ width: fill, height: hug, gap: 12 }, [
              badge("SC26 CDC", "#12345C", "#BFE8FF"),
              badge("Reconstruction", "#163B32", "#C9F4E8"),
            ]),
            t("CDC Reconstruction Weekly Progress", {
              width: wrap(820),
              size: 76,
              bold: true,
              color: COLORS.white,
              fontFamily: DISPLAY_FONT,
            }),
            t("DeltaAI GH200 baseline completed; Delta H200 comparison validated.", {
              width: wrap(760),
              size: 30,
              color: "#D7E3F5",
            }),
            grid({ width: fill, height: fixed(178), columns: [fr(1), fr(1), fr(1)], columnGap: 18 }, [
              metricCard("Realistic batch", "1", "Full-resolution reconstruction", "#7DD3FC", "#102848"),
              metricCard("Step sweep", "5-100", "GH200 full sweep", "#86EFAC", "#102848"),
              metricCard("Compare", "2 systems", "GH200 vs H200", "#FDBA74", "#102848"),
            ]),
            t("Prepared for the week ending Apr 30, 2026", {
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
                path: pathTo("results/2026-04-26-reconstruction/visual_examples_small/comparison_100_0005_0001.jpg"),
                width: fill,
                height: grow(1),
                fit: "cover",
                alt: "Reconstruction visual comparison example",
              }),
              t("Visual check example: reconstructed drone imagery across denoising settings", {
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

function addExperimentDesign(presentation) {
  slideShell(
    presentation,
    "Experiment Structure Is Now Set",
    "The reconstruction task answers one question: how fast can we use the compressed data again?",
    [
      grid({ width: fill, height: grow(1), columns: [fr(1), fr(1), fr(1)], columnGap: 22 }, [
        evidencePanel([
          badge("1", COLORS.blueSoft, COLORS.blue),
          t("Single-image sanity test", { size: 31, bold: true }),
          t("Run one full-resolution image first to verify model loading, inference, metrics, and output paths.", {
            size: 22,
            color: COLORS.muted,
          }),
        ]),
        evidencePanel([
          badge("2", COLORS.tealSoft, COLORS.teal),
          t("Realistic batch-size pilot", { size: 31, bold: true }),
          t("Test batch size under full-image memory pressure before longer sweeps. Current decision: batch size 1.", {
            size: 22,
            color: COLORS.muted,
          }),
        ]),
        evidencePanel([
          badge("3", COLORS.orangeSoft, COLORS.orange),
          t("Repeated parameter sweeps", { size: 31, bold: true }),
          t("Repeat runs and average time, throughput, memory, PSNR, SSIM, and BPP for each setting.", {
            size: 22,
            color: COLORS.muted,
          }),
        ]),
      ]),
      grid({ width: fill, height: fixed(150), columns: [fr(1), fr(1), fr(1), fr(1)], columnGap: 16 }, [
        metricCard("Primary time metric", "sec/image", "Inference time and total time", COLORS.blue),
        metricCard("Throughput", "img/hr", "Same format for Jacob", COLORS.teal),
        metricCard("Resource pressure", "GPU memory", "Peak memory per config", COLORS.orange),
        metricCard("Quality", "PSNR / SSIM", "BPP from fp32 rows", COLORS.green),
      ]),
    ],
    "Source: SC26 experiment design notes and committed reconstruction workflow.",
  );
}

function addPlatformSlide(presentation) {
  const platformRows = [
    ["DeltaAI", "GH200", "bfod-dtai-gh", "Completed full reconstruction sweep"],
    ["Delta", "H200", "bfod-delta-gpu", "Validated quick comparison sweep"],
  ];
  slideShell(
    presentation,
    "Compute Baseline Covers Both Systems",
    "We should describe the current baseline as DeltaAI GH200, then use Delta H200 for hardware comparison.",
    [
      grid({ width: fill, height: grow(1), columns: [fr(1.12), fr(0.88)], columnGap: 28 }, [
        evidencePanel([
          t("Confirmed Access", { size: 30, bold: true }),
          tableRows(
            ["System", "GPU", "Account", "Current role"],
            platformRows,
            [0.75, 0.55, 0.95, 1.6],
            { rowHeight: 58, bodySize: 19 },
          ),
        ]),
        evidencePanel([
          t("Important hardware distinction", { size: 30, bold: true }),
          bullet("DeltaAI uses GH200.", "The 2026-04-26 reconstruction sweep belongs to this baseline.", COLORS.blue),
          bullet("Delta has H200 partitions.", "`gpuH200x8` and `gpuH200x8-interactive` were visible on Apr 28.", COLORS.teal),
          bullet("H100 is not confirmed.", "No H100 partition appeared in the checked Delta `sinfo` output.", COLORS.orange),
        ], COLORS.white),
      ]),
      grid({ width: fill, height: fixed(154), columns: [fr(1), fr(1), fr(1), fr(1)], columnGap: 16 }, [
        metricCard("Delta H200 nodes", "gpue[01-08]", "Observed in `sinfo`", COLORS.teal),
        metricCard("H200 GPU memory", "143.8 GB", "NVIDIA H200 node check", COLORS.blue),
        metricCard("Delta balance", "1076 / 2037", "GPU hours on Apr 28", COLORS.orange),
        metricCard("Runtime check", "PyTorch 2.8", "CUDA available on H200", COLORS.green),
      ]),
    ],
    "Source: repository README, Delta login/resource checks, and NCSA Delta documentation.",
  );
}

function addGh200Baseline(presentation) {
  slideShell(
    presentation,
    "DeltaAI GH200 Baseline Is Complete",
    "The full sweep shows diffusion inference dominates reconstruction time.",
    [
      grid({ width: fill, height: grow(1), columns: [fr(1.22), fr(0.78)], columnGap: 28 }, [
        panel(
          {
            width: fill,
            height: fill,
            fill: COLORS.white,
            line: COLORS.faint,
            borderRadius: 8,
            padding: { x: 18, y: 16 },
          },
          image({
            path: pathTo("results/2026-04-26-reconstruction/plots/plot_time_vs_steps.png"),
            width: fill,
            height: fill,
            fit: "contain",
            alt: "DeltaAI GH200 time versus denoising steps plot",
          }),
        ),
        evidencePanel([
          t("Key GH200 Findings", { size: 30, bold: true }),
          bullet("Batch size is fixed at 1.", "Batch size 2 caused CUDA OOM for full-image reconstruction.", COLORS.red),
          bullet("65-step fp32 profile takes 151.18 s/image.", "Inference alone is 143.80 s/image, or 95.1% of runtime.", COLORS.blue),
          bullet("Reducing steps is the strongest speed lever.", "5-step fp32 is 11.27 s/image; 20-step fp32 is 44.37 s/image.", COLORS.teal),
          bullet("fp16 cuts memory pressure.", "Peak memory drops from about 52.2 GB to 34.2 GB.", COLORS.orange),
        ]),
      ]),
    ],
    "Source: results/2026-04-26-reconstruction.",
  );
}

function addGh200Tradeoff(presentation) {
  slideShell(
    presentation,
    "Speed Gains Come Mostly From Fewer Steps",
    "Quality falls gradually as step count increases, while runtime grows almost linearly.",
    [
      grid({ width: fill, height: grow(1), columns: [fr(1), fr(1)], columnGap: 24 }, [
        panel(
          {
            width: fill,
            height: fill,
            fill: COLORS.white,
            line: COLORS.faint,
            borderRadius: 8,
            padding: { x: 18, y: 16 },
          },
          image({
            path: pathTo("results/2026-04-26-reconstruction/plots/plot_quality_vs_speed.png"),
            width: fill,
            height: fill,
            fit: "contain",
            alt: "DeltaAI GH200 quality versus speed plot",
          }),
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
          image({
            path: pathTo("results/2026-04-26-reconstruction/plots/plot_memory_vs_steps.png"),
            width: fill,
            height: fill,
            fit: "contain",
            alt: "DeltaAI GH200 memory versus denoising steps plot",
          }),
        ),
      ]),
      grid({ width: fill, height: fixed(138), columns: [fr(1), fr(1), fr(1)], columnGap: 16 }, [
        metricCard("Fastest GH200 setting", "5 steps fp16", "10.47 s/image, 343.8 img/hr", COLORS.teal),
        metricCard("Report BPP with", "fp32", "fp16 BPP is invalid in current output", COLORS.blue),
        metricCard("Memory drop with fp16", "34.5%", "52.2 GB to 34.2 GB", COLORS.orange),
      ]),
    ],
    "Source: DeltaAI GH200 step sweep, batch size 1, N=15 per configuration.",
  );
}

function addH200Slide(presentation) {
  const batch1 = h200Batch.find((rowItem) => rowItem.batch_size === 1);
  const batch2 = h200Batch.find((rowItem) => rowItem.batch_size === 2);
  slideShell(
    presentation,
    "Delta H200 Is Runnable and Slightly Faster",
    "A quick H200 sweep confirms the same workflow works on Delta and keeps batch size 1 as the best setting.",
    [
      grid({ width: fill, height: grow(1), columns: [fr(1.18), fr(0.82)], columnGap: 28 }, [
        panel(
          {
            width: fill,
            height: fill,
            fill: COLORS.white,
            line: COLORS.faint,
            borderRadius: 8,
            padding: { x: 18, y: 16 },
          },
          image({
            path: pathTo("results/2026-04-28-h200-reconstruction/plots/plot_time_vs_steps.png"),
            width: fill,
            height: fill,
            fit: "contain",
            alt: "Delta H200 time versus denoising steps plot",
          }),
        ),
        evidencePanel([
          t("H200 Quick Results", { size: 30, bold: true }),
          bullet("20-step fp32: 42.74 s/image.", "Throughput is 84.2 img/hr with 52.2 GB peak memory.", COLORS.blue),
          bullet("65-step fp32: 138.48 s/image.", "Slightly faster than the GH200 65-step fp32 result.", COLORS.teal),
          bullet("Batch size 2 fits but slows down.", `Pilot: batch 1 ${fmt(batch1.avg_inference_sec, 2)} s/image; batch 2 ${fmt(batch2.avg_inference_sec, 2)} s/image.`, COLORS.orange),
          bullet("Decision remains batch size 1.", "This is the realistic setting to share with Jacob for matched experiment design.", COLORS.green),
        ]),
      ]),
    ],
    "Source: results/2026-04-28-h200-reconstruction.",
  );
}

function addComparisonSlide(presentation) {
  const steps = [5, 20, 65];
  const ghFp32 = steps.map((step) => metric(gh200, "fp32", step, "avg_inference_sec"));
  const hFp32 = steps.map((step) => metric(h200, "fp32", step, "avg_inference_sec"));
  const ghFp16 = steps.map((step) => metric(gh200, "fp16", step, "avg_inference_sec"));
  const hFp16 = steps.map((step) => metric(h200, "fp16", step, "avg_inference_sec"));
  const fp32Avg = steps.reduce((sum, step, index) => sum + speedup(ghFp32[index], hFp32[index]), 0) / steps.length;
  const fp16Avg = steps.reduce((sum, step, index) => sum + speedup(ghFp16[index], hFp16[index]), 0) / steps.length;

  slideShell(
    presentation,
    "Matched GH200 vs H200 Comparison",
    "At the shared step counts, H200 improves timing modestly; it does not change the main bottleneck story.",
    [
      grid({ width: fill, height: grow(1), columns: [fr(1.05), fr(0.95)], columnGap: 28 }, [
        evidencePanel([
          chart({
            name: "matched-time-chart",
            chartType: "bar",
            width: fill,
            height: fill,
            config: {
              title: "Inference sec/image, fp32, lower is better",
              categories: steps.map((step) => `${step} steps`),
              series: [
                { name: "DeltaAI GH200", values: ghFp32 },
                { name: "Delta H200", values: hFp32 },
              ],
            },
          }),
        ]),
        evidencePanel([
          t("What Changed on H200?", { size: 30, bold: true }),
          tableRows(
            ["Steps", "GH200 fp32", "H200 fp32", "H200 faster"],
            steps.map((step, index) => [
              step,
              `${fmt(ghFp32[index], 2)}s`,
              `${fmt(hFp32[index], 2)}s`,
              `${fmt(speedup(ghFp32[index], hFp32[index]), 1)}%`,
            ]),
            [0.55, 0.9, 0.9, 0.9],
            { rowHeight: 54, bodySize: 19 },
          ),
          grid({ width: fill, height: fixed(134), columns: [fr(1), fr(1)], columnGap: 16 }, [
            metricCard("Average fp32 gain", `${fmt(fp32Avg, 1)}%`, "H200 over GH200", COLORS.teal, COLORS.tealSoft),
            metricCard("Average fp16 gain", `${fmt(fp16Avg, 1)}%`, "H200 over GH200", COLORS.blue, COLORS.blueSoft),
          ]),
          t("Interpretation: hardware helps a little, but the number of denoising steps remains the dominant runtime lever.", {
            size: 22,
            color: COLORS.muted,
          }),
        ]),
      ]),
    ],
    "Source: GH200 and H200 sweep summary CSVs; matched steps 5, 20, and 65.",
  );
}

function addVisualSlide(presentation) {
  slideShell(
    presentation,
    "Visual Check Supports the Fast Settings",
    "The first GitHub-sized visual inspection did not show a clear failure at low step counts for this example.",
    [
      grid({ width: fill, height: grow(1), columns: [fr(1.25), fr(0.75)], columnGap: 28 }, [
        panel(
          {
            width: fill,
            height: fill,
            fill: COLORS.white,
            line: COLORS.faint,
            borderRadius: 8,
            padding: { x: 16, y: 16 },
          },
          image({
            path: pathTo("results/2026-04-26-reconstruction/visual_examples_small/comparison_100_0005_0001.jpg"),
            width: fill,
            height: fill,
            fit: "contain",
            alt: "Small visual comparison across reconstruction settings",
          }),
        ),
        evidencePanel([
          t("Current Reading", { size: 30, bold: true }),
          bullet("5-step output passed a first visual check.", "This is not a final perceptual study, but it supports testing fewer steps seriously.", COLORS.teal),
          bullet("Metrics stay close across practical settings.", "PSNR and SSIM decline slowly as runtime increases with more steps.", COLORS.blue),
          bullet("Use the same figure style going forward.", "This gives Jacob a matching format for compression plots.", COLORS.orange),
        ]),
      ]),
    ],
    "Source: visual_examples_small/comparison_100_0005_0001.jpg.",
  );
}

function addClosingSlide(presentation) {
  slideShell(
    presentation,
    "This Week's Bottom Line",
    "The reconstruction pipeline is now measured well enough to report, compare, and extend.",
    [
      grid({ width: fill, height: grow(1), columns: [fr(1), fr(1)], columnGap: 28 }, [
        evidencePanel([
          badge("Completed", COLORS.tealSoft, COLORS.teal),
          bullet("DeltaAI GH200 full reconstruction sweep.", "Single-image test, batch pilot, repeated step sweep, fp32/fp16 plots, reports, and visual examples.", COLORS.teal),
          bullet("Delta H200 quick comparison sweep.", "H200 access, runtime, dependencies, batch pilot, and matched 5/20/65 step sweep are documented.", COLORS.teal),
          bullet("GitHub organization by phase.", "Results are stored by date and hardware target for meeting traceability.", COLORS.teal),
        ]),
        evidencePanel([
          badge("Next", COLORS.orangeSoft, COLORS.orange),
          bullet("Run a full H200 sweep if the group wants a final hardware table.", "The quick sweep is enough for direction; a full sweep would match all GH200 step counts.", COLORS.orange),
          bullet("Fix or explain fp16 BPP.", "Use fp32 for compression-ratio reporting until the invalid fp16 BPP is resolved.", COLORS.orange),
          bullet("Test reconstruction optimizations.", "Candidate next knobs: fewer steps, tiled reconstruction, reuse loaded model, and multi-GPU inference.", COLORS.orange),
        ]),
      ]),
      grid({ width: fill, height: fixed(140), columns: [fr(1), fr(1), fr(1)], columnGap: 16 }, [
        metricCard("Recommended batch", "1", "Full-resolution workload", COLORS.blue),
        metricCard("Best speed lever", "steps", "5-step setting is fastest so far", COLORS.teal),
        metricCard("Comparison target", "GH200 vs H200", "H100 not confirmed", COLORS.orange),
      ]),
    ],
    "Source: committed repository results through Apr 30, 2026.",
  );
}

function buildDeck() {
  const presentation = Presentation.create({
    slideSize: { width: SLIDE.width, height: SLIDE.height },
  });

  addCover(presentation);
  addExperimentDesign(presentation);
  addPlatformSlide(presentation);
  addGh200Baseline(presentation);
  addGh200Tradeoff(presentation);
  addH200Slide(presentation);
  addComparisonSlide(presentation);
  addVisualSlide(presentation);
  addClosingSlide(presentation);

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
  fs.mkdirSync(path.dirname(OUTPUT), { recursive: true });
  fs.mkdirSync(PREVIEW_DIR, { recursive: true });

  const presentation = buildDeck();
  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(OUTPUT);
  await renderPreviews(presentation);

  console.log(`Saved PPTX: ${OUTPUT}`);
  console.log(`Saved previews: ${PREVIEW_DIR}`);
}

await main();
