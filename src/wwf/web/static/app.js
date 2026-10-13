// WWF Solver front end. Progressive enhancement only: every page works without this file
// except the review grid's keyboard flow.
(function () {
  "use strict";
  document.documentElement.classList.remove("no-js");
  const RECENT_KEY = "wwf.recentSessions";
  const OPTIONS_KEY = "wwf.options";

  function readJSON(key, fallback) {
    try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch (e) { return fallback; }
  }
  function writeJSON(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch (e) { /* storage blocked */ }
  }

  // --- sending pictures ----------------------------------------------------------------------
  // Pictures are scaled so the long side is at most 2000 px before upload: camera photos stay far
  // under the 10 MB limit, and formats the server can't read (iPhone HEIC) become JPEG.
  const MAX_SIDE = 2000;

  async function prepareImage(blob) {
    try {
      const bitmap = await createImageBitmap(blob); // applies EXIF orientation
      const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
      const keep = scale === 1 && /^image\/(png|jpeg)$/.test(blob.type) && blob.size < 8e6;
      if (keep) return blob;
      const canvas = document.createElement("canvas");
      canvas.width = Math.round(bitmap.width * scale);
      canvas.height = Math.round(bitmap.height * scale);
      canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height);
      const type = blob.type === "image/png" ? "image/png" : "image/jpeg"; // keep screenshots crisp
      return await new Promise((ok) => canvas.toBlob((b) => ok(b || blob), type, 0.92));
    } catch (e) {
      return blob; // the browser can't decode it either: let the server report the problem
    }
  }

  async function sendImage(action, blob, source) {
    const progress = document.getElementById("send-progress");
    const error = document.getElementById("send-error");
    if (progress) progress.hidden = false;
    if (error) error.hidden = true;
    document.querySelectorAll(".image-form, #camera-button").forEach((el) => el.classList.add("busy"));
    try {
      const image = await prepareImage(blob);
      const body = new FormData();
      body.append("image", image, image.type === "image/png" ? "board.png" : "board.jpg");
      body.append("source", source || "auto");
      const res = await fetch(action, { method: "POST", body: body });
      if (res.ok && res.redirected) { window.location.href = res.url; return; }
      const doc = new DOMParser().parseFromString(await res.text(), "text/html");
      const msg = doc.querySelector(".notice.error");
      throw new Error(msg ? msg.textContent.trim() : "Upload failed (" + res.status + ").");
    } catch (e) {
      if (error) { error.textContent = e.message; error.hidden = false; }
      if (progress) progress.hidden = true;
      document.querySelectorAll(".busy").forEach((el) => el.classList.remove("busy"));
    }
  }

  // Any picture form sends as soon as a file is chosen or a photo taken: no extra tap.
  function initImageForms() {
    document.querySelectorAll("form.image-form").forEach((form) => {
      const input = form.querySelector("input[type=file]");
      input.addEventListener("change", () => {
        const source = form.querySelector("input[name=source]");
        if (input.files.length) sendImage(form.action, input.files[0], source && source.value);
        input.value = "";
      });
    });
  }

  // --- upload page ---------------------------------------------------------------------------
  function initUpload() {
    const zone = document.getElementById("dropzone");
    if (!zone) return;
    const fileForm = document.getElementById("file-form");
    ["dragenter", "dragover"].forEach((ev) => zone.addEventListener(ev, (e) => {
      e.preventDefault(); zone.classList.add("drag");
    }));
    ["dragleave", "drop"].forEach((ev) => zone.addEventListener(ev, (e) => {
      e.preventDefault(); zone.classList.remove("drag");
    }));
    zone.addEventListener("drop", (e) => {
      if (e.dataTransfer.files.length) sendImage(fileForm.action, e.dataTransfer.files[0], "auto");
    });
    initViewfinder();

    const recent = readJSON(RECENT_KEY, []);
    if (recent.length) {
      const list = document.getElementById("recent-list");
      recent.forEach((r) => {
        const li = document.createElement("li");
        const a = document.createElement("a");
        a.href = "/review/" + r.id;
        a.textContent = new Date(r.at).toLocaleString();
        li.appendChild(a);
        list.appendChild(li);
      });
      document.getElementById("recent").hidden = false;
    }
  }

  // On a computer, the capture attribute does nothing, so show a live webcam viewfinder instead.
  // Phones keep the native camera app, which takes sharper pictures. Browsers only allow this on
  // https or localhost; elsewhere the button falls back to the file picker.
  async function initViewfinder() {
    const media = navigator.mediaDevices;
    if (!media || !media.getUserMedia || matchMedia("(pointer: coarse)").matches) return;
    let devices = [];
    try { devices = await media.enumerateDevices(); } catch (e) { return; }
    if (!devices.some((d) => d.kind === "videoinput")) return;

    const button = document.getElementById("camera-button");
    const input = button.querySelector("input");
    const finder = document.getElementById("viewfinder");
    const video = document.getElementById("viewfinder-video");
    const form = document.getElementById("camera-form");
    let stream = null;
    const stop = () => {
      if (stream) stream.getTracks().forEach((t) => t.stop());
      stream = null; finder.hidden = true; form.hidden = false;
    };

    button.addEventListener("click", async (e) => {
      e.preventDefault();
      try {
        stream = await media.getUserMedia({
          video: { facingMode: "environment", width: { ideal: 1920 }, height: { ideal: 1080 } },
        });
      } catch (err) {
        input.click(); // camera refused or busy: choose a file instead
        return;
      }
      video.srcObject = stream;
      form.hidden = true; finder.hidden = false;
    });
    document.getElementById("viewfinder-cancel").addEventListener("click", stop);
    document.getElementById("shutter").addEventListener("click", () => {
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth; canvas.height = video.videoHeight;
      canvas.getContext("2d").drawImage(video, 0, 0);
      stop();
      canvas.toBlob((blob) => blob && sendImage(form.action, blob, "photo"), "image/jpeg", 0.92);
    });
  }

  function rememberSession(id) {
    const recent = readJSON(RECENT_KEY, []).filter((r) => r.id !== id);
    recent.unshift({ id: id, at: Date.now() });
    writeJSON(RECENT_KEY, recent.slice(0, 3));
  }

  // --- review page ---------------------------------------------------------------------------
  function initReview() {
    const layout = document.querySelector(".review-layout");
    if (!layout) return;
    rememberSession(layout.dataset.session);

    const board = document.querySelector(".board");
    const cells = Array.from(board.querySelectorAll("input.cell"));
    const size = 15;
    const at = (r, c) => cells[r * size + c];
    const rackCells = Array.from(document.querySelectorAll("input.rack-cell"));
    let lastFocused = null;

    function markBlank(input) {
      input.classList.toggle("blank", /^[a-z]$/.test(input.value) || input.value === "?");
    }
    function move(input, dr, dc) {
      const r = +input.dataset.r, c = +input.dataset.c;
      const nr = Math.min(size - 1, Math.max(0, r + dr));
      const nc = Math.min(size - 1, Math.max(0, c + dc));
      const next = at(nr, nc);
      if (next) { next.focus(); next.select(); }
    }

    cells.forEach((input) => {
      input.addEventListener("focus", () => { lastFocused = input; input.select(); });
      input.addEventListener("keydown", (e) => {
        const arrows = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
        if (arrows[e.key]) {
          e.preventDefault();
          move(input, ...arrows[e.key]);
        } else if (e.key === "Backspace") {
          // Clear this cell and step left; on an empty cell, step left and clear that one.
          e.preventDefault();
          const hadValue = input.value !== "";
          input.value = ""; markBlank(input);
          move(input, 0, -1);
          const prev = document.activeElement;
          if (!hadValue && prev !== input && prev.classList.contains("cell")) {
            prev.value = ""; markBlank(prev);
          }
        } else if (e.key === "Delete") {
          e.preventDefault(); input.value = ""; markBlank(input);
        } else if (/^[a-zA-Z]$/.test(e.key) && !e.metaKey && !e.ctrlKey && !e.altKey) {
          // Typing: uppercase is a tile, lowercase marks a blank; then advance right.
          e.preventDefault();
          input.value = e.key;
          markBlank(input);
          move(input, 0, 1);
        }
      });
      // Mobile keyboards often skip keydown; fall back to the input event.
      input.addEventListener("input", () => {
        input.value = input.value.replace(/[^a-zA-Z]/g, "").slice(-1);
        markBlank(input);
        if (input.value) move(input, 0, 1);
      });
    });

    rackCells.forEach((input, i) => {
      input.addEventListener("focus", () => { lastFocused = input; input.select(); });
      input.addEventListener("input", () => {
        const v = input.value.slice(-1);
        input.value = v === "?" ? "?" : v.replace(/[^a-zA-Z]/g, "").toUpperCase();
        markBlank(input);
        if (input.value && rackCells[i + 1]) rackCells[i + 1].focus();
      });
      input.addEventListener("keydown", (e) => {
        if (e.key === "Backspace" && !input.value && rackCells[i - 1]) {
          e.preventDefault(); rackCells[i - 1].value = ""; rackCells[i - 1].focus();
        } else if (e.key === "ArrowLeft" && rackCells[i - 1]) {
          rackCells[i - 1].focus();
        } else if (e.key === "ArrowRight" && rackCells[i + 1]) {
          rackCells[i + 1].focus();
        }
      });
    });

    // Phones make lowercase awkward: this flips the last focused board cell between tile and blank.
    const toggle = document.getElementById("blank-toggle");
    if (toggle) {
      toggle.hidden = false;
      toggle.addEventListener("mousedown", (e) => e.preventDefault()); // keep the cell focused
      toggle.addEventListener("click", () => {
        const input = lastFocused;
        if (!input || !input.value || input.classList.contains("rack-cell")) return;
        input.value = input.value === input.value.toUpperCase()
          ? input.value.toLowerCase() : input.value.toUpperCase();
        markBlank(input);
        input.focus();
      });
    }
  }

  // --- results page --------------------------------------------------------------------------
  function initResults() {
    const table = document.querySelector("table.moves");
    if (!table) return;
    table.addEventListener("click", (e) => {
      const link = e.target.closest("a");
      if (!link) return;
      table.querySelectorAll("tr.selected").forEach((tr) => tr.classList.remove("selected"));
      link.closest("tr").classList.add("selected");
    });
    // On a phone the preview sits below the table: bring it into view once it has loaded.
    document.body.addEventListener("htmx:afterSwap", (e) => {
      if (e.detail.target.id === "preview" && window.matchMedia("(max-width: 899px)").matches) {
        e.detail.target.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
    // Clicking anywhere on a row opens its preview, not just the word.
    table.querySelectorAll("tbody tr").forEach((tr) => {
      tr.style.cursor = "pointer";
      tr.addEventListener("click", (e) => {
        if (e.target.closest("a")) return;
        tr.querySelector("a").click();
      });
    });
  }

  // --- options page --------------------------------------------------------------------------
  function initOptions() {
    const form = document.getElementById("options-form");
    if (!form) return;
    form.addEventListener("submit", () => {
      const data = {};
      new FormData(form).forEach((v, k) => { data[k] = v; });
      writeJSON(OPTIONS_KEY, data);
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    initImageForms();
    initUpload();
    initReview();
    initResults();
    initOptions();
  });
})();
