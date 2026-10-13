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

  // --- upload page ---------------------------------------------------------------------------
  function initUpload() {
    const form = document.getElementById("upload-form");
    if (!form) return;
    const zone = document.getElementById("dropzone");
    const input = form.querySelector("input[type=file]");
    const progress = document.getElementById("upload-progress");
    const text = zone.querySelector(".dropzone-text");

    input.addEventListener("change", () => {
      if (input.files.length) text.textContent = input.files[0].name;
    });
    ["dragenter", "dragover"].forEach((ev) => zone.addEventListener(ev, (e) => {
      e.preventDefault(); zone.classList.add("drag");
    }));
    ["dragleave", "drop"].forEach((ev) => zone.addEventListener(ev, (e) => {
      e.preventDefault(); zone.classList.remove("drag");
    }));
    zone.addEventListener("drop", (e) => {
      if (e.dataTransfer.files.length) {
        input.files = e.dataTransfer.files;
        input.dispatchEvent(new Event("change"));
      }
    });
    form.addEventListener("submit", () => {
      progress.hidden = false;
      form.querySelector("button[type=submit]").disabled = true;
    });

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
    initUpload();
    initReview();
    initResults();
    initOptions();
  });
})();
