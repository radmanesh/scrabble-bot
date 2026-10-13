// WWF Solver front end. Progressive enhancement only: every page works without this file
// except the review grid's keyboard flow.
(function () {
  "use strict";
  document.documentElement.classList.remove("no-js");
  const OPTIONS_KEY = "wwf.options";

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

  // With stay=true (fixture collection with the webcam), a successful upload refreshes the banner
  // in place instead of leaving the page, so the camera stays open for the next picture.
  async function sendImage(action, blob, source, stay) {
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
      // Next turn on the review page: send the board as it is now, unsaved edits included, so
      // the server keeps those tiles and only reads the new ones from the picture.
      const boardForm = document.getElementById("board-form");
      if (boardForm && action.includes("/game/")) {
        new FormData(boardForm).forEach((v, k) => { if (k.startsWith("c-")) body.append(k, v); });
      }
      const res = await fetch(action, { method: "POST", body: body });
      const doc = new DOMParser().parseFromString(await res.text(), "text/html");
      if (res.ok && res.redirected) {
        const banner = doc.querySelector(".fixture-banner");
        const current = document.querySelector(".fixture-banner");
        if (!stay || !banner || !current) { window.location.href = res.url; return; }
        current.replaceWith(banner);
        if (progress) progress.hidden = true;
        document.querySelectorAll(".busy").forEach((el) => el.classList.remove("busy"));
        return;
      }
      const msg = doc.querySelector(".notice.error");
      throw new Error(msg ? msg.textContent.trim() : "Upload failed (" + res.status + ").");
    } catch (e) {
      if (error) { error.textContent = e.message; error.hidden = false; }
      if (progress) progress.hidden = true;
      document.querySelectorAll(".busy").forEach((el) => el.classList.remove("busy"));
    }
  }

  // A chosen file or a captured photo is shown first, with "Read the board" and "Choose
  // another": nothing is sent, and nothing in the game changes, until it is confirmed.
  let staged = null;
  function stageImage(blob, action, source) {
    const box = document.getElementById("import-preview");
    if (!box) { sendImage(action, blob, source); return; }  // a page with no preview area
    const img = box.querySelector("img");
    if (img.src.startsWith("blob:")) URL.revokeObjectURL(img.src);
    img.src = URL.createObjectURL(blob);
    staged = { blob, action, source };
    const error = document.getElementById("send-error");
    if (error) error.hidden = true;
    box.hidden = false;
    box.querySelector("[data-go]").focus();
  }
  function initImportPreview() {
    const box = document.getElementById("import-preview");
    if (!box) return;
    box.querySelector("[data-go]").addEventListener("click", () => {
      if (staged) sendImage(staged.action, staged.blob, staged.source);
    });
    box.querySelector("[data-cancel]").addEventListener("click", () => {
      const img = box.querySelector("img");
      if (img.src.startsWith("blob:")) URL.revokeObjectURL(img.src);
      img.removeAttribute("src");
      staged = null; box.hidden = true;
      const error = document.getElementById("send-error");
      if (error) error.hidden = true;
      const again = document.getElementById("camera-button");
      if (again) again.focus();
    });
  }

  // Any picture form stages its file as soon as it is chosen or a photo taken.
  function initImageForms() {
    document.querySelectorAll("form.image-form").forEach((form) => {
      const input = form.querySelector("input[type=file]");
      input.addEventListener("change", () => {
        const source = form.querySelector("input[name=source]");
        if (input.files.length) stageImage(input.files[0], form.action, source && source.value);
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
      if (e.dataTransfer.files.length) stageImage(e.dataTransfer.files[0], fileForm.action, "auto");
    });
  }

  // On a computer, the capture attribute does nothing, so show a live webcam viewfinder instead.
  // Phones keep the native camera app, which takes sharper pictures. Browsers only allow this on
  // https or localhost; elsewhere the button falls back to the file picker. Whether a camera
  // exists is found out by asking for it on click: Safari lists no devices before permission.
  function initViewfinder() {
    const media = navigator.mediaDevices;
    if (!media || !media.getUserMedia || matchMedia("(pointer: coarse)").matches) return;

    const button = document.getElementById("camera-button");
    const finder = document.getElementById("viewfinder");
    if (!button || !finder) return; // this page has no camera button
    const input = button.querySelector("input");
    const frame = finder.querySelector(".viewfinder-frame");
    const video = document.getElementById("viewfinder-video");
    const countdown = document.getElementById("countdown");
    const form = document.getElementById("camera-form");
    const stay = finder.dataset.stay === "yes"; // fixture collection: keep the camera open
    let stream = null;
    let timer = null;
    const stop = () => {
      clearInterval(timer); countdown.hidden = true;
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
        // Refused, busy or no camera. The click's user activation is gone after the await, so
        // the file picker can't be opened from here: say what to do instead.
        const error = document.getElementById("send-error");
        if (error) {
          error.textContent = "Couldn't open the camera (" + (err.name || "unavailable") +
            "). Allow camera access for this site, or use \u201cupload a screenshot\u201d below.";
          error.hidden = false;
        }
        return;
      }
      // A laptop's camera faces the user: mirror the preview like a selfie camera so the phone
      // moves the way you expect. The captured frame itself is never mirrored.
      const facing = stream.getVideoTracks()[0].getSettings().facingMode;
      video.classList.toggle("mirrored", facing !== "environment");
      video.srcObject = stream;
      form.hidden = true; finder.hidden = false;
    });

    function capture() {
      if (!stream || !video.videoWidth) return;
      const canvas = document.createElement("canvas");
      canvas.width = video.videoWidth; canvas.height = video.videoHeight;
      canvas.getContext("2d").drawImage(video, 0, 0);
      frame.classList.remove("flash"); void frame.offsetWidth; frame.classList.add("flash");
      if (!stay) stop();
      canvas.toBlob((blob) => {
        if (!blob) return;
        if (stay) sendImage(form.action, blob, "photo", stay);  // collecting pictures: no pause
        else stageImage(blob, form.action, "photo");
      }, "image/jpeg", 0.92);
    }
    function captureIn(seconds) {
      clearInterval(timer);
      let left = seconds;
      countdown.textContent = left; countdown.hidden = false;
      timer = setInterval(() => {
        left -= 1;
        if (left > 0) { countdown.textContent = left; return; }
        clearInterval(timer); countdown.hidden = true; capture();
      }, 1000);
    }

    document.getElementById("shutter").addEventListener("click", capture);
    document.getElementById("shutter-timer").addEventListener("click", () => captureIn(3));
    document.getElementById("viewfinder-cancel").addEventListener("click", stop);
    document.addEventListener("keydown", (e) => {
      if (finder.hidden || e.repeat) return;
      if (e.key === " " || e.key === "Enter") { e.preventDefault(); capture(); }
      if (e.key === "Escape") stop();
    });
  }

  // --- game page: the editable grid ----------------------------------------------------------
  function initReview() {
    const layout = document.querySelector(".game");
    if (!layout) return;

    const board = document.querySelector("#board-form .board");
    if (!board) return;
    const cells = Array.from(board.querySelectorAll("input.cell"));
    const size = 15;
    const at = (r, c) => cells[r * size + c];
    const rackCells = Array.from(document.querySelectorAll("input.rack-cell"));
    let lastFocused = null;

    function markBlank(input) {
      input.classList.toggle("blank", /^[a-z]$/.test(input.value) || input.value === "?");
      describeCell(input);
    }
    function move(input, dr, dc) {
      const r = +input.dataset.r, c = +input.dataset.c;
      const nr = Math.min(size - 1, Math.max(0, r + dr));
      const nc = Math.min(size - 1, Math.max(0, c + dc));
      const next = at(nr, nc);
      if (next) { next.focus(); next.select(); }
    }

    // "Go to this square" in the picture review focuses that square on the board.
    document.querySelectorAll("a.goto-square").forEach((a) => a.addEventListener("click", (e) => {
      const target = at(+a.dataset.r, +a.dataset.c);
      if (!target) return;
      e.preventDefault();
      target.scrollIntoView({ block: "center" });
      target.focus();
    }));

    // Clicking a square fills in where the opponent's typed word starts.
    const oppRow = document.querySelector(".opp-type input[name=row]");
    const oppCol = document.querySelector(".opp-type input[name=col]");
    function pointOpponentAt(input) {
      if (oppRow && oppCol) {
        oppRow.value = +input.dataset.r + 1; oppCol.value = +input.dataset.c + 1;
        oppRow.dispatchEvent(new Event("input", { bubbles: true })); // refresh the live preview
      }
    }

    // One tab stop for the whole board: Tab enters at the last square used and leaves on the next
    // Tab; arrow keys do the moving. Without JavaScript every square stays an ordinary tab stop.
    let rover = cells.find((c) => !c.readOnly && !c.disabled) || cells[0];
    cells.forEach((c) => { c.tabIndex = -1; });
    rover.tabIndex = 0;
    function setRover(el) { if (el === rover) return; rover.tabIndex = -1; rover = el; el.tabIndex = 0; }
    const zoomButton = document.getElementById("board-zoom");
    const card = document.getElementById("board-form");
    if (zoomButton) {
      const tools = zoomButton.closest(".board-tools");
      if (tools) tools.hidden = false;
      const setZoom = (on) => {
        card.classList.toggle("zoomed", on);
        zoomButton.setAttribute("aria-pressed", String(on));
        zoomButton.textContent = on ? "Smaller board" : "Larger board";
        if (on && lastFocused) lastFocused.scrollIntoView({ block: "nearest", inline: "nearest" });
      };
      zoomButton.addEventListener("click", () => setZoom(!card.classList.contains("zoomed")));
      card.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && card.classList.contains("zoomed")) { setZoom(false); zoomButton.focus(); }
      });
    }

    cells.forEach((input) => {
      input.addEventListener("focus", () => {
        lastFocused = input; setRover(input); input.select(); pointOpponentAt(input);
        if (card && card.classList.contains("zoomed")) input.scrollIntoView({ block: "nearest", inline: "nearest" });
      });
      input.addEventListener("keydown", (e) => {
        const arrows = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
        if (arrows[e.key]) {
          e.preventDefault();
          move(input, ...arrows[e.key]);
        } else if (e.key === "Home" || e.key === "End") {
          e.preventDefault();
          move(input, 0, e.key === "Home" ? -size : size);  // to the start or end of the row
        } else if (input.readOnly) {
          // a square from a confirmed turn: read-only, only navigation works
          if (!e.metaKey && !e.ctrlKey && e.key.length === 1) e.preventDefault();
        } else if (e.key === "Backspace") {
          // Clear this cell and step left; on an empty cell, step left and clear that one.
          e.preventDefault();
          const hadValue = input.value !== "";
          input.value = ""; markBlank(input);
          move(input, 0, -1);
          const prev = document.activeElement;
          if (!hadValue && prev !== input && prev.classList.contains("cell") && !prev.readOnly) {
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
        if (input.readOnly) return;
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
        if (!input || !input.value || input.readOnly || input.classList.contains("rack-cell")) return;
        input.value = input.value === input.value.toUpperCase()
          ? input.value.toLowerCase() : input.value.toUpperCase();
        markBlank(input);
        input.focus();
      });
    }
  }



  // What a square says to assistive technology, kept in step with what it shows.
  function describeCell(input) {
    const where = input.dataset.where;
    if (!where) return;
    const v = input.value;
    const kind = /[a-z]/.test(v) || input.classList.contains("blank") ? "blank tile " : "tile ";
    let text;
    if (input.classList.contains("proposed")) text = `${where}, proposed ${kind}${v.toUpperCase()}`;
    else if (!v) text = `${where}, empty`;
    else if (input.readOnly) text = `${where}, ${kind}${v.toUpperCase()}, confirmed, locked`;
    else text = `${where}, ${kind}${v.toUpperCase()}, not confirmed yet`;
    input.setAttribute("aria-label", text);
  }

  // --- previews drawn on the board -----------------------------------------------------------
  // A previewed word or suggestion is shown on the board squares without being part of the
  // form: previewed tiles are disabled, so the board form never submits them, and clear() puts
  // every square back exactly as it was.
  function boardOverlay() {
    const board = document.querySelector("#board-form .board");
    let touched = [];
    const cellAt = (r, c) => board && board.querySelector(`input.cell[data-r="${r}"][data-c="${c}"]`);
    // anything the server drew (no JavaScript) is taken off first, so clear() can undo it
    if (board) {
      board.querySelectorAll("input.cell.proposed:disabled").forEach((el) => {
        el.value = ""; el.disabled = false; el.classList.remove("proposed", "blank");
      });
      board.querySelectorAll(".cell.crossing").forEach((el) => el.classList.remove("crossing"));
    }
    function remember(el) {
      if (!touched.some((t) => t.el === el)) touched.push({ el, value: el.value, wasDisabled: el.disabled });
    }
    function clear() {
      touched.forEach(({ el, value, wasDisabled }) => {
        el.value = value; el.disabled = wasDisabled;
        el.classList.remove("proposed", "crossing", "start-marker");
        if (!value) el.classList.remove("blank");
        describeCell(el);
      });
      touched = [];
    }
    function show(cells, start) {
      clear();
      cells.forEach((c) => {
        const el = cellAt(c.r, c.c);
        if (!el) return;
        remember(el);
        if (c.new) {
          el.value = c.letter; el.disabled = true;
          el.classList.add("proposed");
          el.classList.toggle("blank", !!c.blank);
        } else {
          el.classList.add("crossing");
        }
        describeCell(el);
      });
      if (start) {
        const el = cellAt(start[0] - 1, start[1] - 1);
        if (el) { remember(el); el.classList.add("start-marker"); }
      }
    }
    return { show, clear };
  }

  // --- game page: the move composer ----------------------------------------------------------
  // Types a word, chooses where it starts and which way, and draws it on the board with its score
  // before anything is recorded. The preview only changes what the squares show: previewed tiles
  // are disabled, so the board form never submits them, and Clear puts the squares back.
  function initComposer() {
    const form = document.getElementById("composer-form");
    if (!form) return;
    const game = document.getElementById("composer").dataset.game;
    const word = document.getElementById("composer-word");
    const row = document.getElementById("composer-row");
    const col = document.getElementById("composer-col");
    const dir = document.getElementById("composer-direction");
    const status = document.getElementById("composer-status");
    const force = document.getElementById("composer-force");
    const letters = document.getElementById("blank-letters");
    const overlay = boardOverlay();
    const clearOverlay = overlay.clear;
    const show = overlay.show;
    let timer = null, seq = 0;
    function say(html) { status.innerHTML = html; }
    const esc = (t) => String(t).replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[ch]);

    async function refresh() {
      const mine = ++seq;
      if (!word.value.trim() || !row.value || !col.value) {
        clearOverlay(); force.hidden = true;
        say('<p class="hint small">Type the word and where it starts to see it on the board and its score.</p>');
        return;
      }
      const q = new URLSearchParams({ word: word.value, row: row.value, col: col.value, direction: dir.value });
      let data;
      try {
        const res = await fetch(`/game/${game}/compose/preview.json?${q}`);
        data = await res.json();
      } catch (e) {
        return;  // offline: the server still checks when it is recorded
      }
      if (mine !== seq) return;  // a newer request is on its way
      show(data.cells || [], data.start);
      const parts = [];
      if (data.problems && data.problems.length) {
        parts.push("<ul class=\"problems\">" + data.problems.map((p) => `<li>${esc(p)}</li>`).join("") + "</ul>");
      }
      if (data.points !== null && data.points !== undefined) {
        const words = (data.words || []).map((w) => `${esc(w.word)} ${w.points}`).join(" + ");
        parts.push(`<p><b>${data.points} points</b> (${data.tiles} new tile${data.tiles === 1 ? "" : "s"})` +
          (words ? `: ${words}` : "") + (data.bingo ? ` + bingo ${data.bingo}` : "") + "</p>");
      }
      say(parts.join("") || "<p class=\"hint small\">Looks fine.</p>");
      force.hidden = !data.forceable;
    }
    function later() { clearTimeout(timer); timer = setTimeout(refresh, 250); }

    // Blank tiles are set with one button per letter of the word, not by typing lowercase by
    // accident: typed letters are capitalised, and a letter keeps its blank flag only while it is
    // still the same letter in the same place. The server still accepts lowercase as a blank.
    let flags = [...word.value.trim()].map((ch) => ch !== ch.toUpperCase());
    let prev = word.value.trim().toUpperCase();
    function write() {
      word.value = [...prev].map((ch, i) => (flags[i] ? ch.toLowerCase() : ch)).join("");
    }
    function drawLetters() {
      letters.textContent = "";
      [...prev].forEach((ch, i) => {
        if (!/[A-Z]/.test(ch)) return;
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = ch;
        b.setAttribute("aria-pressed", String(!!flags[i]));
        b.setAttribute("aria-label", `Letter ${i + 1}, ${ch}${flags[i] ? ", blank tile" : ""}: press to ${flags[i] ? "make it a normal tile" : "make it a blank"}`);
        b.addEventListener("click", () => { flags[i] = !flags[i]; write(); drawLetters(); later(); });
        letters.appendChild(b);
      });
    }
    function typed() {
      const next = word.value.trim().toUpperCase();
      flags = [...next].map((ch, i) => (prev[i] === ch ? !!flags[i] : false));
      prev = next;
      write(); drawLetters(); later();
    }

    word.addEventListener("input", typed);
    [row, col].forEach((el) => el.addEventListener("input", later));
    form.querySelectorAll(".dir-btn").forEach((btn) => btn.addEventListener("click", () => {
      dir.value = btn.dataset.direction;
      form.querySelectorAll(".dir-btn").forEach((b) => {
        b.classList.toggle("active", b === btn);
        b.setAttribute("aria-pressed", String(b === btn));
      });
      later();
    }));
    document.getElementById("composer-cancel").addEventListener("click", () => {
      clearOverlay();
      word.value = ""; prev = ""; flags = []; row.value = ""; col.value = ""; force.hidden = true;
      drawLetters();
      say('<p class="hint small">Type the word and where it starts to see it on the board and its score.</p>');
      word.focus();
    });
    // the server-rendered preview (no JavaScript) is already drawn: take it over cleanly
    drawLetters();
    if (word.value && row.value && col.value) refresh();
  }


  // --- game page: choosing a suggestion ------------------------------------------------------
  // Choosing a move draws it on the board and updates the details beside it. Only Record this move
  // changes the game; without JavaScript the same choice is a link that reloads the page.
  function initSuggestions() {
    const panel = document.getElementById("suggestions");
    const data = document.getElementById("suggestions-data");
    if (!panel || !data) return;
    let moves;
    try { moves = JSON.parse(data.textContent); } catch (e) { return; }
    if (!moves.length) return;
    const overlay = boardOverlay();
    const game = panel.dataset.game;
    const rows = Array.from(panel.querySelectorAll("tbody tr[data-n]"));
    const text = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };

    function choose(n) {
      const m = moves[n];
      if (!m) return;
      overlay.show(m.cells, null);
      rows.forEach((tr) => {
        const on = +tr.dataset.n === n;
        tr.classList.toggle("selected", on);
        const a = tr.querySelector("a");
        if (a) { if (on) a.setAttribute("aria-current", "true"); else a.removeAttribute("aria-current"); }
      });
      text("sel-word", m.word); text("sel-score", m.score);
      const words = m.words.map((w) => `${w.word} ${w.points}`).join(" + ") + (m.bingo ? ` + bingo ${m.bingo}` : "");
      text("sel-detail", `${m.direction === "across" ? "Across" : "Down"} from row ${m.start[0]}, column ${m.start[1]} · ${words} · Tiles left on your rack: ${m.leave || "none"}`);
      document.getElementById("record-form").action = `/game/${game}/record/${n}`;
      text("sel-note", `Previewing ${m.word} for ${m.score} points, shown dashed on the board. Not recorded yet.`);
      history.replaceState(null, "", `?do=find&move=${n}`);
    }
    rows.forEach((tr) => {
      const n = +tr.dataset.n;
      tr.style.cursor = "pointer";
      tr.addEventListener("click", (e) => { e.preventDefault(); choose(n); });
    });
    const more = document.getElementById("show-more");
    if (more) more.addEventListener("click", (e) => {
      e.preventDefault();
      rows.forEach((tr) => { tr.hidden = false; tr.classList.remove("more"); });
      more.parentElement.hidden = true;
    });
    const picked = rows.find((tr) => tr.classList.contains("selected"));
    choose(picked ? +picked.dataset.n : 0);
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

  // Forms that destroy something ask first (data-confirm="question").
  function initConfirms() {
    document.querySelectorAll("form[data-confirm]").forEach((form) => {
      form.addEventListener("submit", (e) => {
        if (!window.confirm(form.dataset.confirm)) e.preventDefault();
      });
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    initConfirms();
    initImageForms();
    initImportPreview();
    initViewfinder();
    initUpload();
    initReview();
    initComposer();
    initSuggestions();
    initResults();
    initOptions();
  });
})();
