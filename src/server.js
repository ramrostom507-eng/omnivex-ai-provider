/**
 * OMNIVEX AI Provider — Cloudflare Workers
 * OpenAI-compatible API gateway for Google Gemini and Groq.
 *
 * Existing src/server.js is intentionally preserved as the Node/Express fallback.
 */

const VERSION = "2.0.1";
const DEFAULT_GEMINI_MODEL = "gemini-2.5-flash";
const DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile";
const MAX_BODY_BYTES = 1_000_000;
const MAX_MESSAGES = 100;
const MAX_MESSAGE_CHARS = 100_000;
const DEFAULT_TIMEOUT_MS = 45_000;
const RATE_WINDOW_MS = 60_000;
const RATE_MAX_REQUESTS = 60;

// Best-effort per-isolate limiter. For strict production limits, add Cloudflare
// rate limiting rules or Durable Objects; this in-memory map is not global.
const rateBuckets = new Map();

function json(data, status = 200, headers = {}) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
      ...headers,
    },
  });
}

function apiError(message, status = 400, type = "invalid_request_error", code = null) {
  return json({ error: { message, type, ...(code ? { code } : {}) } }, status);
}

function allowedOrigins(env) {
  const raw = env.CORS_ORIGIN || "*";
  if (raw.trim() === "*") return ["*"];
  return raw.split(",").map((s) => s.trim()).filter(Boolean);
}

function corsHeaders(request, env) {
  const origin = request.headers.get("Origin") || "";
  const allowed = allowedOrigins(env);
  const originAllowed = allowed.includes("*") || allowed.includes(origin);
  const headers = {
    "access-control-allow-methods": "GET, POST, OPTIONS",
    "access-control-allow-headers": "Authorization, Content-Type, X-Request-ID",
    "access-control-max-age": "86400",
    "vary": "Origin",
  };
  if (originAllowed && origin) {
    headers["access-control-allow-origin"] = allowed.includes("*") ? "*" : origin;
  } else if (allowed.includes("*")) {
    headers["access-control-allow-origin"] = "*";
  }
  return headers;
}

function withCors(response, request, env) {
  const headers = new Headers(response.headers);
  for (const [key, value] of Object.entries(corsHeaders(request, env))) {
    headers.set(key, value);
  }
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

function checkOrigin(request, env) {
  const origin = request.headers.get("Origin");
  if (!origin) return true; // server-to-server clients commonly omit Origin
  const allowed = allowedOrigins(env);
  return allowed.includes("*") || allowed.includes(origin);
}

function constantTimeEqual(a, b) {
  if (typeof a !== "string" || typeof b !== "string" || a.length !== b.length) return false;
  let mismatch = 0;
  for (let i = 0; i < a.length; i++) mismatch |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return mismatch === 0;
}

function authorized(request, env) {
  if (!env.OMNIVEX_API_KEY) return false;
  const header = request.headers.get("Authorization") || "";
  const match = header.match(/^Bearer\s+(.+)$/i);
  return Boolean(match && constantTimeEqual(match[1].trim(), env.OMNIVEX_API_KEY));
}

function clientKey(request) {
  // Cloudflare supplies this header at the edge. It is used only for best-effort
  // throttling and is not logged or returned.
  return request.headers.get("CF-Connecting-IP") || "unknown";
}

function checkRateLimit(request) {
  const now = Date.now();
  const key = clientKey(request);
  let bucket = rateBuckets.get(key);
  if (!bucket || now - bucket.start >= RATE_WINDOW_MS) {
    bucket = { start: now, count: 0 };
  }
  bucket.count += 1;
  rateBuckets.set(key, bucket);

  // Bound memory growth in long-lived isolates.
  if (rateBuckets.size > 5_000) {
    for (const [k, v] of rateBuckets) {
      if (now - v.start >= RATE_WINDOW_MS) rateBuckets.delete(k);
    }
  }
  return {
    allowed: bucket.count <= RATE_MAX_REQUESTS,
    remaining: Math.max(0, RATE_MAX_REQUESTS - bucket.count),
    resetSeconds: Math.max(1, Math.ceil((bucket.start + RATE_WINDOW_MS - now) / 1000)),
  };
}

async function readJson(request) {
  const declaredLength = Number(request.headers.get("content-length") || 0);
  if (declaredLength > MAX_BODY_BYTES) {
    throw Object.assign(new Error("حجم الطلب أكبر من الحد المسموح (1 ميغابايت)."), { status: 413 });
  }
  const text = await request.text();
  if (new TextEncoder().encode(text).length > MAX_BODY_BYTES) {
    throw Object.assign(new Error("حجم الطلب أكبر من الحد المسموح (1 ميغابايت)."), { status: 413 });
  }
  try {
    return JSON.parse(text);
  } catch {
    throw Object.assign(new Error("Invalid JSON request body"), { status: 400 });
  }
}

function validateChatRequest(body) {
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    throw Object.assign(new Error("يجب إرسال جسم طلب JSON صالح."), { status: 400 });
  }
  if (body.stream === true) {
    throw Object.assign(new Error("البث المباشر stream غير مدعوم حالياً؛ أرسل stream=false أو احذف الحقل."), { status: 400 });
  }
  if (!Array.isArray(body.messages) || body.messages.length === 0 || body.messages.length > MAX_MESSAGES) {
    throw Object.assign(new Error(`يجب أن يحتوي messages على 1 إلى ${MAX_MESSAGES} رسالة.`), { status: 400 });
  }

  const messages = body.messages.map((m, i) => {
    if (!m || typeof m !== "object" || !["system", "user", "assistant"].includes(m.role)) {
      throw Object.assign(new Error(`دور الرسالة رقم ${i + 1} غير صالح. الأدوار المدعومة: system, user, assistant.`), { status: 400 });
    }
    let content = m.content;
    if (typeof content !== "string") {
      // Support the common OpenAI text-part array shape.
      if (Array.isArray(content)) {
        content = content
          .filter((part) => part && part.type === "text" && typeof part.text === "string")
          .map((part) => part.text)
          .join("\n");
      } else {
        throw Object.assign(new Error(`محتوى الرسالة رقم ${i + 1} يجب أن يكون نصاً.`), { status: 400 });
      }
    }
    if (!content.trim()) {
      throw Object.assign(new Error(`الرسالة رقم ${i + 1} فارغة.`), { status: 400 });
    }
    if (content.length > MAX_MESSAGE_CHARS) {
      throw Object.assign(new Error(`الرسالة رقم ${i + 1} تتجاوز الحد المسموح.`), { status: 400 });
    }
    return { role: m.role, content };
  });

  return {
    messages,
    requestedModel: typeof body.model === "string" ? body.model.trim() : "",
    temperature: Number.isFinite(body.temperature) ? Math.min(2, Math.max(0, body.temperature)) : 0.7,
    maxTokens: Number.isInteger(body.max_tokens) ? Math.min(8192, Math.max(1, body.max_tokens)) : 2048,
  };
}

function toGeminiContents(messages) {
  const systemText = messages.filter((m) => m.role === "system").map((m) => m.content).join("\n\n");
  const contents = messages
    .filter((m) => m.role !== "system")
    .map((m) => ({
      role: m.role === "assistant" ? "model" : "user",
      parts: [{ text: m.content }],
    }));

  // Gemini expects alternating user/model turns. Merge consecutive same-role
  // messages to avoid provider-side invalid-turn errors.
  const merged = [];
  for (const item of contents) {
    const previous = merged[merged.length - 1];
    if (previous && previous.role === item.role) {
      previous.parts[0].text += "\n\n" + item.parts[0].text;
    } else {
      merged.push(item);
    }
  }
  if (merged.length && merged[0].role === "model") {
    merged.unshift({ role: "user", parts: [{ text: "تابع المحادثة التالية." }] });
  }
  return { systemText, contents: merged };
}

function extractGeminiText(data) {
  return (data?.candidates?.[0]?.content?.parts || [])
    .map((part) => typeof part.text === "string" ? part.text : "")
    .join("")
    .trim();
}

async function fetchWithTimeout(url, options, timeoutMs = DEFAULT_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort("timeout"), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function callGemini(messages, env, model, temperature = 0.7, maxTokens = 2048) {
  if (!env.GEMINI_API_KEY) throw new Error("مفتاح Gemini غير مضبوط على Cloudflare.");
  const { systemText, contents } = toGeminiContents(messages);
  if (!contents.length) throw new Error("أضف رسالة مستخدم واحدة على الأقل.");

  const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent?key=${encodeURIComponent(env.GEMINI_API_KEY)}`;
  const payload = {
    contents,
    generationConfig: { temperature, maxOutputTokens: maxTokens },
  };
  if (systemText) payload.systemInstruction = { parts: [{ text: systemText }] };

  const response = await fetchWithTimeout(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = data?.error?.message || `Gemini API returned HTTP ${response.status}`;
    throw Object.assign(new Error(message), { provider: "gemini", status: response.status });
  }
  const text = extractGeminiText(data);
  if (!text) {
    const reason = data?.promptFeedback?.blockReason || data?.candidates?.[0]?.finishReason || "empty_response";
    throw Object.assign(new Error(`لم يُرجع Gemini نصاً صالحاً (${reason}).`), { provider: "gemini" });
  }
  return { text, model, provider: "gemini" };
}

async function callGroq(messages, env, model, temperature, maxTokens) {
  if (!env.GROQ_API_KEY) throw new Error("مفتاح Groq غير مضبوط على Cloudflare.");
  const response = await fetchWithTimeout("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "authorization": `Bearer ${env.GROQ_API_KEY}`,
    },
    body: JSON.stringify({
      model,
      messages,
      temperature,
      max_tokens: maxTokens,
      stream: false,
    }),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = data?.error?.message || `Groq API returned HTTP ${response.status}`;
    throw Object.assign(new Error(message), { provider: "groq", status: response.status });
  }
  const text = data?.choices?.[0]?.message?.content;
  if (typeof text !== "string" || !text.trim()) {
    throw Object.assign(new Error("Groq أعاد استجابة فارغة."), { provider: "groq" });
  }
  return { text: text.trim(), model: data.model || model, provider: "groq", usage: data.usage };
}

function modelChoice(requested, env) {
  const geminiModel = env.GEMINI_MODEL || DEFAULT_GEMINI_MODEL;
  const groqModel = env.GROQ_MODEL || DEFAULT_GROQ_MODEL;
  if (!requested) return { provider: "auto", model: "" };

  const lower = requested.toLowerCase();
  if (lower.startsWith("gemini:")) {
    return { provider: "gemini", model: requested.slice(7).trim() || geminiModel };
  }
  if (lower.startsWith("gemini/")) {
    return { provider: "gemini", model: requested.slice(7).trim() || geminiModel };
  }
  if (lower.startsWith("google/")) {
    return { provider: "gemini", model: requested.slice(7).trim() || geminiModel };
  }
  if (lower.startsWith("groq:")) {
    return { provider: "groq", model: requested.slice(5).trim() || groqModel };
  }
  if (lower.startsWith("groq/")) {
    return { provider: "groq", model: requested.slice(5).trim() || groqModel };
  }
  if (lower === "auto" || lower === "omnivex-auto") return { provider: "auto", model: "" };
  // Recognize the configured default names even if they don't carry a prefix.
  if (requested === geminiModel) return { provider: "gemini", model: requested };
  if (requested === groqModel) return { provider: "groq", model: requested };
  // OpenAI-compatible callers often send a provider model ID. Treat unknown IDs
  // as Groq IDs; use "gemini:<model>" to select a specific Gemini model.
  return { provider: "groq", model: requested };
}

function openAIResponse(result, requestId) {
  return {
    id: `chatcmpl-${crypto.randomUUID()}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: result.model,
    choices: [{
      index: 0,
      message: { role: "assistant", content: result.text },
      finish_reason: "stop",
    }],
    ...(result.usage ? { usage: result.usage } : {}),
    system_fingerprint: `omnivex-${VERSION}`,
    ...(requestId ? { request_id: requestId } : {}),
  };
}

async function handleChat(request, env) {
  if (!authorized(request, env)) {
    return apiError("مفتاح OMNIVEX API غير صحيح أو غير مضبوط.", 401, "authentication_error", "invalid_api_key");
  }
  const rate = checkRateLimit(request);
  if (!rate.allowed) {
    return json({ error: { message: "تم تجاوز عدد الطلبات المسموح مؤقتاً. حاول بعد دقيقة.", type: "rate_limit_error" } }, 429, {
      "retry-after": String(rate.resetSeconds),
      "x-ratelimit-remaining": "0",
    });
  }

  let body;
  try {
    body = await readJson(request);
    const input = validateChatRequest(body);
    const choice = modelChoice(input.requestedModel, env);
    let result;

    if (choice.provider === "gemini") {
      result = await callGemini(input.messages, env, choice.model || env.GEMINI_MODEL || DEFAULT_GEMINI_MODEL, input.temperature, input.maxTokens);
    } else if (choice.provider === "groq") {
      result = await callGroq(input.messages, env, choice.model || env.GROQ_MODEL || DEFAULT_GROQ_MODEL, input.temperature, input.maxTokens);
    } else {
      // Automatic mode: prefer Groq, then Gemini. Only use a configured provider.
      const attempts = [];
      if (env.GROQ_API_KEY) {
        try {
          result = await callGroq(input.messages, env, env.GROQ_MODEL || DEFAULT_GROQ_MODEL, input.temperature, input.maxTokens);
        } catch (error) { attempts.push(`Groq: ${error.message}`); }
      }
      if (!result && env.GEMINI_API_KEY) {
        try {
          result = await callGemini(input.messages, env, env.GEMINI_MODEL || DEFAULT_GEMINI_MODEL, input.temperature, input.maxTokens);
        } catch (error) { attempts.push(`Gemini: ${error.message}`); }
      }
      if (!result) {
        throw new Error(attempts.join(" | ") || "لم يتم ضبط أي مزود ذكاء اصطناعي. أضف GEMINI_API_KEY أو GROQ_API_KEY.");
      }
    }

    return json(openAIResponse(result, request.headers.get("X-Request-ID")), 200, {
      "x-ratelimit-remaining": String(rate.remaining),
    });
  } catch (error) {
    const status = error.status && Number.isInteger(error.status) ? error.status : 502;
    const message = error.name === "AbortError" ? "انتهت مهلة الاتصال بمزود الذكاء الاصطناعي. حاول مجدداً." : (error.message || "حدث خطأ غير متوقع.");
    return apiError(message, status, status >= 500 ? "provider_error" : "invalid_request_error");
  }
}

async function route(request, env) {
  const url = new URL(request.url);
  const path = url.pathname.replace(/\/+$/, "") || "/";
  const method = request.method.toUpperCase();

  if (method === "OPTIONS") return new Response(null, { status: 204 });
  if (!checkOrigin(request, env)) return apiError("هذا النطاق غير مسموح له باستخدام واجهة API.", 403, "cors_error");

  if (method === "GET" && path === "/") {
    return json({
      name: "OMNIVEX AI Provider",
      description: "Arabic-first OpenAI-compatible AI gateway",
      version: VERSION,
      status: "online",
      endpoints: ["/health", "/v1/models", "/v1/chat/completions"],
    });
  }

  if (method === "GET" && path === "/health") {
    return json({
      status: "ok",
      service: "omnivex-ai-provider",
      version: VERSION,
      providers: {
        gemini: Boolean(env.GEMINI_API_KEY),
        groq: Boolean(env.GROQ_API_KEY),
      },
      authentication: Boolean(env.OMNIVEX_API_KEY),
      timestamp: new Date().toISOString(),
    });
  }

  if (method === "GET" && path === "/v1/models") {
    if (!authorized(request, env)) return apiError("مفتاح OMNIVEX API غير صحيح أو غير مضبوط.", 401, "authentication_error", "invalid_api_key");
    const models = [
      { id: "omnivex-auto", object: "model", owned_by: "omnivex", description: "Automatic provider selection with fallback" },
      { id: env.GROQ_MODEL || DEFAULT_GROQ_MODEL, object: "model", owned_by: "groq" },
      { id: env.GEMINI_MODEL || DEFAULT_GEMINI_MODEL, object: "model", owned_by: "google" },
    ];
    return json({ object: "list", data: models });
  }

  if (method === "POST" && path === "/v1/chat/completions") {
    return handleChat(request, env);
  }

  if (path.startsWith("/v1/")) return apiError("المسار المطلوب غير موجود.", 404, "not_found_error");
  return apiError("المسار غير موجود.", 404, "not_found_error");
}

export default {
  async fetch(request, env) {
    try {
      const response = await route(request, env);
      return withCors(response, request, env);
    } catch (error) {
      const response = apiError(error?.message || "Internal server error", 500, "server_error");
      return withCors(response, request, env);
    }
  },
};
