
const API_URL = "http://127.0.0.1:8000/api";

const chatArea = document.getElementById("chatArea");
const welcome = document.getElementById("welcome");
const messages = document.getElementById("messages");
const historyList = document.getElementById("historyList");
const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const sidebar = document.getElementById("sidebar");
const overlay = document.getElementById("mobileOverlay");


// Token counter elements
const totalTokensElement = document.getElementById("totalTokens");
const inputTokensElement = document.getElementById("inputTokens");
const outputTokensElement = document.getElementById("outputTokens");
const detailTotalTokensElement =
  document.getElementById("detailTotalTokens");
const apiCallsElement = document.getElementById("apiCalls");
const databaseReadsElement = document.getElementById("databaseReads");
const totalDbReadsElement = document.getElementById("totalDbReads");

const tokenDetailsButton =
  document.getElementById("tokenDetailsButton");
const tokenPopover = document.getElementById("tokenPopover");
const closeTokenDetailsButton =
  document.getElementById("closeTokenDetails");

// Search elements & state
const historySearchInput = document.getElementById("historySearchInput");
const clearSearchButton = document.getElementById("clearSearchButton");
let allSessionsCache = [];
let searchDebounceTimer = null;
let currentSearchQuery = "";

// Attachment elements & state
const attachButton = document.getElementById("attachButton");
const fileInput = document.getElementById("fileInput");
const attachmentPreview = document.getElementById("attachmentPreview");
const attachmentChipIcon = document.getElementById("attachmentChipIcon");
const attachmentChipName = document.getElementById("attachmentChipName");
const attachmentChipSize = document.getElementById("attachmentChipSize");
const removeAttachmentBtn = document.getElementById("removeAttachmentBtn");
let selectedAttachmentFile = null;

// Keep track of the displayed session usage.
let currentTokenUsage = {
  input_tokens: 0,
  output_tokens: 0,
  total_tokens: 0,
  api_calls: 0,
  database_reads: 0,
};


function renderTokenUsage(usage = {}) {
  const input = Number(usage.input_tokens ?? 0);
  const output = Number(usage.output_tokens ?? 0);
  const total = Number(
    usage.total_tokens ?? input + output
  );
  const calls = Number(usage.api_calls ?? 0);
  const reads = Number(
    usage.database_reads ?? usage.db_reads ?? 0
  );

  totalTokensElement.textContent = total.toLocaleString();
  inputTokensElement.textContent = input.toLocaleString();
  outputTokensElement.textContent = output.toLocaleString();
  detailTotalTokensElement.textContent = total.toLocaleString();
  apiCallsElement.textContent = calls.toLocaleString();

  if (databaseReadsElement) {
    databaseReadsElement.textContent = reads.toLocaleString();
  }
  if (totalDbReadsElement) {
    totalDbReadsElement.textContent = reads.toLocaleString();
  }

  currentTokenUsage = {
    input_tokens: input,
    output_tokens: output,
    total_tokens: total,
    api_calls: calls,
    database_reads: reads,
  };
}


async function loadTokenUsage(sessionId) {
  if (!sessionId) {
    renderTokenUsage();
    return;
  }

  const data = await apiRequest(
    `/sessions/${encodeURIComponent(sessionId)}/usage`
  );

  // Ignore a late response if the user switched chats.
  if (currentSessionId !== sessionId) return;

  const usageData = {
    ...(data.token_usage || {}),
    database_reads:
      data.token_usage?.database_reads ??
      data.database_reads ??
      0,
  };

  renderTokenUsage(usageData);
}

// Token details popup
tokenDetailsButton.addEventListener("click", () => {
  tokenPopover.hidden = !tokenPopover.hidden;
});

closeTokenDetailsButton.addEventListener("click", () => {
  tokenPopover.hidden = true;
});


let currentSessionId = null;
let isSending = false;
renderTokenUsage();
tokenPopover.hidden = true;

function createElement(tag, className, text = "") {
  const element = document.createElement(tag);
  element.className = className;
  element.textContent = text;
  return element;
}

function scrollToBottom() {
  chatArea.scrollTop = chatArea.scrollHeight;
}

function formatFileSize(bytes) {
  const size = Number(bytes);
  if (!size || size <= 0) return "0 B";
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

function clearAttachment() {
  selectedAttachmentFile = null;
  if (fileInput) fileInput.value = "";
  if (attachmentPreview) attachmentPreview.hidden = true;
}

function setAttachment(file) {
  if (!file) return;

  const ext = (file.name.split(".").pop() || "").toLowerCase();
  if (ext !== "pdf" && ext !== "txt") {
    alert("Only PDF (.pdf) and Plain Text (.txt) files are supported.");
    clearAttachment();
    return;
  }

  const maxBytes = 10 * 1024 * 1024; // 10MB
  if (file.size > maxBytes) {
    alert(`File exceeds 10 MB limit (${formatFileSize(file.size)}). Please choose a smaller document.`);
    clearAttachment();
    return;
  }

  if (file.size === 0) {
    alert("The selected file is empty (0 bytes). Please choose a valid document.");
    clearAttachment();
    return;
  }

  selectedAttachmentFile = file;
  if (attachmentChipName) attachmentChipName.textContent = file.name;
  if (attachmentChipSize) attachmentChipSize.textContent = formatFileSize(file.size);
  if (attachmentChipIcon) attachmentChipIcon.textContent = ext === "pdf" ? "📕" : "📄";
  if (attachmentPreview) attachmentPreview.hidden = false;

  messageInput.focus();
}

async function uploadFile(file, sessionId) {
  const formData = new FormData();
  formData.append("file", file);
  if (sessionId) {
    formData.append("session_id", sessionId);
  }

  let response;
  try {
    response = await fetch(`${API_URL}/upload`, {
      method: "POST",
      body: formData,
    });
  } catch {
    throw new Error("Cannot connect to the backend. Make sure FastAPI is running.");
  }

  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || "Failed to upload document.");
  }
  return data;
}

function renderMessage(role, content, attachment = null) {
  welcome.hidden = true;

  const message = createElement(
    "div",
    `message ${role === "user" ? "user" : "assistant"}`
  );

  const avatar = createElement(
    "div",
    "avatar",
    role === "user" ? "Y" : "✦"
  );

  const body = createElement("div", "message-content");

  const roleLabel = createElement(
    "div",
    "message-role",
    role === "user" ? "You" : "Nova AI"
  );

  body.append(roleLabel);

  if (attachment && attachment.filename) {
    const badge = createElement("div", "message-attachment");
    const icon = createElement(
      "span",
      "message-attachment-icon",
      (attachment.file_type || "").toLowerCase() === "pdf" ? "📕" : "📄"
    );
    const name = createElement("span", "message-attachment-name", attachment.filename);
    name.title = attachment.filename;
    const size = createElement(
      "span",
      "message-attachment-size",
      formatFileSize(attachment.file_size)
    );
    badge.append(icon, name, size);
    body.append(badge);
  }

  const text = createElement("div", "message-text");

  if (role === "assistant") {
    // Convert Markdown into HTML.
    const parsedHTML = marked.parse(content);
    text.innerHTML = DOMPurify.sanitize(parsedHTML);

    // Find all code blocks inside the AI response.
    text.querySelectorAll("pre code").forEach((code) => {
      const pre = code.parentElement;

      // Detect the programming language from Markdown.
      const languageClass = [...code.classList].find((name) =>
        name.startsWith("language-")
      );

      const language = languageClass
        ? languageClass.replace("language-", "")
        : "code";

      // Add syntax highlighting.
      if (window.Prism) {
        Prism.highlightElement(code);
      }

      // Create a wrapper for the code block.
      const wrapper = document.createElement("div");
      wrapper.className = "code-block-wrapper";

      const toolbar = document.createElement("div");
      toolbar.className = "code-toolbar";

      const label = document.createElement("span");
      label.className = "code-language";
      label.textContent = language.toUpperCase();

      const copyButton = document.createElement("button");
      copyButton.className = "copy-code-button";
      copyButton.type = "button";
      copyButton.textContent = "Copy";

      copyButton.addEventListener("click", async () => {
        // textContent copies the actual code, not highlighted HTML.
        const codeText = code.textContent;

        try {
          await navigator.clipboard.writeText(codeText);
          copyButton.textContent = "Copied!";

          setTimeout(() => {
            copyButton.textContent = "Copy";
          }, 1500);
        } catch {
          copyButton.textContent = "Copy failed";
        }
      });

      toolbar.append(label, copyButton);

      // Replace the original pre with our styled wrapper.
      pre.parentNode.insertBefore(wrapper, pre);
      wrapper.append(toolbar, pre);
    });

  } else {
    // Keep user messages as plain text.
    text.textContent = content;
  }

  body.append(text);
  message.append(avatar, body);
  messages.appendChild(message);

  scrollToBottom();
  return message;
}

function showTyping(statusText = "Thinking...") {
  const message = createElement("div", "message assistant");
  const avatar = createElement("div", "avatar", "✦");
  const body = createElement("div", "message-content");
  const label = createElement("div", "message-role", "Nova AI");
  const typing = createElement("div", "typing", statusText);

  body.append(label, typing);
  message.append(avatar, body);
  messages.appendChild(message);
  scrollToBottom();

  return message;
}

async function apiRequest(path, options = {}) {
  let response;

  try {
    response = await fetch(`${API_URL}${path}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
    });
  } catch {
    throw new Error(
      "Cannot connect to the backend. Make sure FastAPI is running."
    );
  }

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(data.detail || "Something went wrong.");
  }

  return data;
}

function renderSessionList(sessions = [], isSearch = false) {
  historyList.replaceChildren();

  if (!sessions || sessions.length === 0) {
    const emptyState = createElement("div", "history-empty");
    if (isSearch) {
      const heading = createElement("span", "", "No matching conversations found");
      const sub = createElement("small", "", "Try a different keyword or topic");
      emptyState.append(heading, sub);
    } else {
      const heading = createElement("span", "", "No conversations yet");
      const sub = createElement("small", "", "Start a new chat to begin");
      emptyState.append(heading, sub);
    }
    historyList.appendChild(emptyState);
    return;
  }

  sessions.forEach((session) => {
    const row = createElement("div", "history-row");

    const hasSnippet = isSearch && Boolean(session.matching_snippet);
    const openButton = createElement(
      "button",
      `history-item ${
        session.session_id === currentSessionId ? "active" : ""
      } ${hasSnippet ? "has-snippet" : ""}`
    );

    openButton.dataset.sessionId = session.session_id;
    const titleText = session.title || "New conversation";
    openButton.title = titleText;

    if (hasSnippet) {
      const titleSpan = createElement("span", "history-title", titleText);
      const snippetSpan = createElement("small", "search-snippet", session.matching_snippet);
      openButton.append(titleSpan, snippetSpan);
    } else {
      openButton.textContent = titleText;
    }

    openButton.addEventListener("click", () => {
      openSession(session.session_id);
    });

    const deleteButton = createElement(
      "button",
      "delete-chat-button",
      "×"
    );

    deleteButton.type = "button";
    deleteButton.title = "Delete chat";
    deleteButton.setAttribute("aria-label", "Delete chat");

    deleteButton.addEventListener("click", async (event) => {
      event.stopPropagation();

      const confirmed = window.confirm(
        `Delete "${session.title || "this chat"}" permanently?`
      );

      if (!confirmed) return;

      try {
        await apiRequest(
          `/sessions/${encodeURIComponent(session.session_id)}`,
          { method: "DELETE" }
        );

        // Clear the visible conversation if it was deleted.
        if (currentSessionId === session.session_id) {
          currentSessionId = null;
          messages.replaceChildren();
          welcome.hidden = false;

          // Clear deleted session's token usage.
          renderTokenUsage();
          tokenPopover.hidden = true;
        }

        const activeQuery = historySearchInput ? historySearchInput.value.trim() : "";
        if (activeQuery) {
          await performSearch(activeQuery);
        } else {
          await loadSessions();
        }
      } catch (error) {
        alert(`Could not delete chat: ${error.message}`);
      }
    });

    const renameButton = createElement(
      "button",
      "rename-chat-button",
      "✎"
    );

    renameButton.type = "button";
    renameButton.title = "Rename chat";
    renameButton.setAttribute("aria-label", "Rename chat");

    renameButton.addEventListener("click", async (event) => {
      event.stopPropagation();

      const newTitle = window.prompt(
        "Enter a new chat title:",
        session.title || ""
      );

      if (newTitle === null) return;

      const title = newTitle.trim();

      if (!title) {
        alert("Chat title cannot be empty.");
        return;
      }

      if (title.length > 60) {
        alert("Chat title cannot exceed 60 characters.");
        return;
      }

      try {
        await apiRequest(
          `/sessions/${encodeURIComponent(session.session_id)}/title`,
          {
            method: "PATCH",
            body: JSON.stringify({ title }),
          }
        );

        const activeQuery = historySearchInput ? historySearchInput.value.trim() : "";
        if (activeQuery) {
          await performSearch(activeQuery);
        } else {
          await loadSessions();
        }
      } catch (error) {
        alert(`Could not rename chat: ${error.message}`);
      }
    });

    row.append(openButton, renameButton, deleteButton);
    historyList.appendChild(row);
  });
}

async function loadSessions() {
  try {
    const data = await apiRequest("/sessions");
    allSessionsCache = data.sessions || [];

    // If search is active, keep current search view but update active highlight
    if (historySearchInput && historySearchInput.value.trim()) {
      document.querySelectorAll(".history-item").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.sessionId === currentSessionId);
      });
      return;
    }

    renderSessionList(allSessionsCache, false);
  } catch (error) {
    console.error("Could not load chat history:", error.message);
  }
}

async function performSearch(rawQuery) {
  const query = rawQuery.trim();
  currentSearchQuery = query;

  if (!query) {
    if (clearSearchButton) clearSearchButton.hidden = true;
    renderSessionList(allSessionsCache, false);
    return;
  }

  if (clearSearchButton) clearSearchButton.hidden = false;

  try {
    const data = await apiRequest(
      `/sessions/search?q=${encodeURIComponent(query)}`
    );

    // Guard against race conditions if query changed while in flight
    if (currentSearchQuery !== query) return;

    renderSessionList(data.sessions || [], true);
  } catch (error) {
    console.error("Could not search chat history:", error.message);
  }
}

function handleSearchInput() {
  const rawValue = historySearchInput.value;
  if (clearSearchButton) {
    clearSearchButton.hidden = rawValue.length === 0;
  }

  clearTimeout(searchDebounceTimer);
  searchDebounceTimer = setTimeout(() => {
    performSearch(rawValue);
  }, 250);
}

function clearSearch() {
  if (historySearchInput) {
    historySearchInput.value = "";
    historySearchInput.focus();
  }
  if (clearSearchButton) {
    clearSearchButton.hidden = true;
  }
  currentSearchQuery = "";
  renderSessionList(allSessionsCache, false);
}


async function openSession(sessionId, fromCommand = false) {
  if (isSending && !fromCommand) return;

  try {
    const session = await apiRequest(
      `/sessions/${encodeURIComponent(sessionId)}`
    );

    currentSessionId = session.session_id;
    messages.replaceChildren();

    session.messages.forEach((message) => {
      renderMessage(message.role, message.content, message.attachment);
    });

    clearAttachment();

    welcome.hidden = session.messages.length > 0;

    // Load the saved token totals for this conversation.
    await loadTokenUsage(session.session_id);

    await loadSessions();
    closeMobileSidebar();
  } catch (error) {
    alert(error.message);
  }
}



function newChat() {
  if (isSending) return;

  currentSessionId = null;
  messages.replaceChildren();
  clearAttachment();
  welcome.hidden = false;

  // Reset token usage for the new conversation.
  renderTokenUsage();
  tokenPopover.hidden = true;

  messageInput.value = "";
  messageInput.style.height = "auto";
  messageInput.focus();

  loadSessions();
  closeMobileSidebar();
}



async function sendMessage(text) {
  const message = text.trim();

  if ((!message && !selectedAttachmentFile) || isSending) return;

  // 1. Detect session commands before calling OpenAI (only when no file is attached)
  if (message && !selectedAttachmentFile) {
    // Detect commands to open an existing session.
    const openMatch = message.match(
      /\b(?:open|show|get|load)\s+session\s+id\s+(.+)/i
    );

    if (openMatch) {
      const sessionId = openMatch[1]
        .trim()
        .replace(/^["']|["']$/g, "")
        .replace(/[.,!?]+$/, "");

      console.log("OPEN SESSION COMMAND DETECTED:", message);
      console.log("SESSION ID:", sessionId);

      const uuidPattern =
        /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

      if (!uuidPattern.test(sessionId)) {
        alert(
          "Invalid session ID. Copy the complete session_id from MongoDB Compass."
        );
        return;
      }

      isSending = true;
      sendButton.disabled = true;
      messageInput.disabled = true;
      if (attachButton) attachButton.disabled = true;

      try {
        await openSession(sessionId, true);
      } catch (error) {
        alert(`Could not open session: ${error.message}`);
      } finally {
        isSending = false;
        sendButton.disabled = false;
        messageInput.disabled = false;
        if (attachButton) attachButton.disabled = false;
        messageInput.focus();
      }

      return;
    }

    const deleteCommand = message.match(
      /\bdelete\s+session\s+id\s+(.+)/i
    );

    if (deleteCommand) {
      const sessionId = deleteCommand[1]
        .trim()
        .replace(/^["']|["']$/g, "")
        .replace(/[.,!?]+$/, "");

      // Validate the session ID format.
      const uuidPattern =
        /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

      if (!uuidPattern.test(sessionId)) {
        alert(
          "Invalid session ID. Copy the complete session_id from MongoDB Compass."
        );
        return;
      }

      // Prevent another send while deletion is running.
      isSending = true;
      sendButton.disabled = true;
      messageInput.disabled = true;
      if (attachButton) attachButton.disabled = true;

      try {
        // Get session details to verify that it exists.
        const session = await apiRequest(
          `/sessions/${encodeURIComponent(sessionId)}`
        );

        const sessionTitle = session.title || sessionId;

        // Ask for confirmation before permanent deletion.
        const confirmed = window.confirm(
          `Permanently delete "${sessionTitle}"?\n\n` +
          `Session ID: ${sessionId}\n\n` +
          "This will remove the session and its messages from MongoDB."
        );

        if (!confirmed) return;

        // Delete the session through the backend API.
        await apiRequest(
          `/sessions/${encodeURIComponent(sessionId)}`,
          { method: "DELETE" }
        );

        // Clear the current conversation if it was deleted.
        if (currentSessionId === sessionId) {
          currentSessionId = null;
          messages.replaceChildren();
          clearAttachment();
          welcome.hidden = false;

          // Clear the deleted session's token usage.
          renderTokenUsage();
          tokenPopover.hidden = true;
        }

        // Refresh the sidebar.
        await loadSessions();

        alert("Session and its messages deleted successfully.");
      } catch (error) {
        alert(`Could not delete session: ${error.message}`);
      } finally {
        isSending = false;
        sendButton.disabled = false;
        messageInput.disabled = false;
        if (attachButton) attachButton.disabled = false;
        messageInput.focus();
      }

      return;
    }
  }

  // 2. Normal AI chat flow (supports optional document attachment).
  isSending = true;
  sendButton.disabled = true;
  messageInput.disabled = true;
  if (attachButton) attachButton.disabled = true;

  const fileToSend = selectedAttachmentFile;
  clearAttachment();

  const promptText = message || (fileToSend ? `Please summarize and explain the contents of ${fileToSend.name}.` : "");

  let tempAttachment = null;
  if (fileToSend) {
    tempAttachment = {
      filename: fileToSend.name,
      file_type: (fileToSend.name.split(".").pop() || "").toLowerCase(),
      file_size: fileToSend.size,
    };
  }

  renderMessage("user", promptText, tempAttachment);
  messageInput.value = "";
  messageInput.style.height = "auto";

  let typing = showTyping(fileToSend ? "Processing document & thinking..." : "Thinking...");

  try {
    let uploadedFileId = null;

    if (fileToSend) {
      const uploadResult = await uploadFile(fileToSend, currentSessionId);
      uploadedFileId = uploadResult.file_id;
      if (!currentSessionId && uploadResult.session_id) {
        currentSessionId = uploadResult.session_id;
      }
    }

    const result = await apiRequest("/chat", {
      method: "POST",
      body: JSON.stringify({
        message: promptText,
        session_id: currentSessionId,
        file_id: uploadedFileId,
      }),
    });

    currentSessionId = result.session_id;

    if (typing) typing.remove();

    renderMessage("assistant", result.reply);

    // Optimistically reflect database reads from chat response
    if (result.database_reads !== undefined) {
      if (totalDbReadsElement) {
        totalDbReadsElement.textContent = Number(result.database_reads).toLocaleString();
      }
      if (databaseReadsElement) {
        databaseReadsElement.textContent = Number(result.database_reads).toLocaleString();
      }
    }

    // Refresh cumulative token and database reads usage from MongoDB.
    try {
      await loadTokenUsage(result.session_id);
    } catch (usageError) {
      console.error("Could not load token usage:", usageError);
    }

    await loadSessions();

  } catch (error) {
    if (typing) typing.remove();

    renderMessage(
      "assistant",
      `Sorry, something went wrong.\n\n${error.message}\n\nPlease check the backend and try again.`
    );
  } finally {
    isSending = false;
    sendButton.disabled = false;
    messageInput.disabled = false;
    if (attachButton) attachButton.disabled = false;
    messageInput.focus();
  }
}


chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(messageInput.value);
});

messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

messageInput.addEventListener("input", () => {
  messageInput.style.height = "auto";
  messageInput.style.height =
    `${Math.min(messageInput.scrollHeight, 180)}px`;
});

document.getElementById("newChat").addEventListener("click", newChat);

document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => {
    sendMessage(button.dataset.prompt || "");
  });
});

document.getElementById("menuButton").addEventListener("click", () => {
  sidebar.classList.add("open");
  overlay.classList.add("show");
});

document.getElementById("closeSidebar").addEventListener(
  "click",
  closeMobileSidebar
);

overlay.addEventListener("click", closeMobileSidebar);

function closeMobileSidebar() {
  sidebar.classList.remove("open");
  overlay.classList.remove("show");
}

if (historySearchInput) {
  historySearchInput.addEventListener("input", handleSearchInput);
  historySearchInput.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      clearSearch();
    }
  });
}

if (clearSearchButton) {
  clearSearchButton.addEventListener("click", clearSearch);
}

// File Attachment Event Listeners
if (attachButton && fileInput) {
  attachButton.addEventListener("click", () => {
    if (isSending) return;
    fileInput.click();
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) {
      setAttachment(fileInput.files[0]);
    }
  });
}

if (removeAttachmentBtn) {
  removeAttachmentBtn.addEventListener("click", () => {
    clearAttachment();
    messageInput.focus();
  });
}

loadSessions();
