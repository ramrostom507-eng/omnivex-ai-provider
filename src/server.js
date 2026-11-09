import { httpServerHandler } from "cloudflare:node";
import express from "express";
import cors from "cors";
import helmet from "helmet";
import dotenv from "dotenv";
import { GoogleGenerativeAI } from "@google/generative-ai";
import Groq from "groq-sdk";
import { randomUUID, timingSafeEqual } from "node:crypto";

dotenv.config();

const app = express();
const PORT = Number(process.env.PORT) || 3000;

app.disable("x-powered-by");

// Render runs the app behind a reverse proxy.
// Set TRUST_PROXY=false only when running directly without a trusted proxy.
app.set(
  "trust proxy",
  process.env.TRUST_PROXY === "false" ? false : 1
);

app.use(helmet());

// CORS configuration
const allowedOrigins = (process.env.CORS_ORIGIN || "")
  .split(",")
  .map(origin => origin.trim().replace(/\/$/, ""))
  .filter(Boolean);

app.use(cors({
  origin(origin, callback) {
    // Allow server-to-server requests with no Origin header.
    if (!origin) {
      return callback(null, true);
    }

    const normalizedOrigin = origin.replace(/\/$/, "");

    if (allowedOrigins.includes(normalizedOrigin)) {
      return callback(null, true);
    }

    return callback(new Error("Origin not allowed"));
  },
  methods: ["GET", "POST", "OPTIONS"],
  allowedHeaders: ["Content-Type", "Authorization"],
  maxAge: 86400
}));

app.use(express.json({
  limit: "1mb",
  strict: true
}));

// AI providers
const gemini = process.env.GEMINI_API_KEY
  ? new GoogleGenerativeAI(process.env.GEMINI_API_KEY)
  : null;

const groq = process.env.GROQ_API_KEY
  ? new Groq({ apiKey: process.env.GROQ_API_KEY })
  : null;

const GEMINI_MODEL =
  process.env.GEMINI_MODEL || "gemini-2.5-flash";

const GROQ_MODEL =
  process.env.GROQ_MODEL || "llama-3.3-70b-versatile";

// Rate limiter
// This is an in-memory limiter for a single running instance.
const requests = new Map();

const RATE_LIMIT = 60;
const WINDOW_MS = 60_000;
const MAX_TRACKED_IPS = 20_000;

let rateLimitChecks = 0;

function rateLimit(req, res, next) {
  const now = Date.now();
  const ip = req.ip || req.socket.remoteAddress || "unknown";

  // Periodically remove expired records to prevent unbounded growth.
  rateLimitChecks++;

  if (rateLimitChecks % 100 === 0) {
    for (const [key, value] of requests) {
      if (now - value.start >= WINDOW_MS) {
        requests.delete(key);
      }
    }
  }

  let record = requests.get(ip);

  if (!record || now - record.start >= WINDOW_MS) {
    record = {
      start: now,
      count: 0
    };
  }

  record.count++;
  requests.set(ip, record);

  // Keep the map bounded if many different IPs appear.
  while (requests.size > MAX_TRACKED_IPS) {
    const oldestKey = requests.keys().next().value;

    if (oldestKey === undefined) {
      break;
    }

    requests.delete(oldestKey);
  }

  if (record.count > RATE_LIMIT) {
    return res.status(429).json({
      error: {
        message: "Too many requests. Try again later.",
        type: "rate_limit_error"
      }
    });
  }

  return next();
}

// API authentication
function authenticate(req, res, next) {
  const apiKey = process.env.OMNIVEX_API_KEY || "";

  if (!apiKey.trim()) {
    return res.status(503).json({
      error: {
        message: "OMNIVEX API authentication is not configured",
        type: "configuration_error"
      }
    });
  }

  const authorization = req.headers.authorization || "";

  const suppliedKey = authorization.startsWith("Bearer ")
    ? authorization.slice(7).trim()
    : "";

  const expected = Buffer.from(apiKey);
  const supplied = Buffer.from(suppliedKey);

  const valid =
    expected.length === supplied.length &&
    timingSafeEqual(expected, supplied);

  if (!valid) {
    return res.status(401).json({
      error: {
        message: "Invalid API key",
        type: "authentication_error"
      }
    });
  }

  return next();
}

// Request validation errors are different from provider failures.
class ValidationError extends Error {
  constructor(message) {
    super(message);
    this.name = "ValidationError";
  }
}

// Validate and normalize messages
function normalizeMessages(messages) {
  if (
    !Array.isArray(messages) ||
    messages.length < 1 ||
    messages.length > 100
  ) {
    throw new ValidationError(
      "Messages must contain between 1 and 100 entries"
    );
  }

  let totalLength = 0;

  const cleanMessages = messages.map(message => {
    if (
      !message ||
      !["system", "user", "assistant"].includes(message.role)
    ) {
      throw new ValidationError("Invalid message role");
    }

    if (typeof message.content !== "string") {
      throw new ValidationError(
        "Message content must be text"
      );
    }

    const content = message.content.trim();

    if (!content || content.length > 50_000) {
      throw new ValidationError("Invalid message content");
    }

    totalLength += content.length;

    if (totalLength > 100_000) {
      throw new ValidationError(
        "Total message size is too large"
      );
    }

    return {
      role: message.role,
      content
    };
  });

  return cleanMessages;
}

// Gemini provider
async function askGemini(messages, modelName, temperature) {
  if (!gemini) {
    throw new Error("Gemini provider is not configured");
  }

  const systemMessages = messages
    .filter(message => message.role === "system")
    .map(message => message.content);

  const conversation = messages
    .filter(message => message.role !== "system")
    .map(message => {
      const label =
        message.role === "assistant" ? "Assistant" : "User";

      return `${label}: ${message.content}`;
    })
    .join("\n\n");

  if (!conversation.trim()) {
    throw new ValidationError(
      "At least one user or assistant message is required"
    );
  }

  const model = gemini.getGenerativeModel({
    model: modelName,
    systemInstruction:
      systemMessages.join("\n\n") || undefined,
    generationConfig: {
      temperature
    }
  });

  const result = await model.generateContent(conversation);
  const answer = result.response.text();

  if (typeof answer !== "string" || !answer.trim()) {
    throw new Error("Empty response from Gemini");
  }

  return answer.trim();
}

// Groq provider
async function askGroq(messages, modelName, temperature) {
  if (!groq) {
    throw new Error("Groq provider is not configured");
  }

  const result = await groq.chat.completions.create({
    model: modelName,
    messages,
    temperature
  });

  const answer = result.choices?.[0]?.message?.content;

  if (typeof answer !== "string" || !answer.trim()) {
    throw new Error("Empty response from Groq");
  }

  return answer.trim();
}

// Main endpoint
app.get("/", (req, res) => {
  return res.json({
    name: "OMNIVEX AI",
    description: "Independent Arabic-first AI provider",
    version: "1.3.0",
    status: "running"
  });
});

// Health check
app.get("/health", (req, res) => {
  return res.json({
    status: "healthy",
    service: "OMNIVEX AI Provider",
    version: "1.3.0",
    providers: {
      gemini: Boolean(gemini),
      groq: Boolean(groq)
    }
  });
});

// Models endpoint
app.get("/v1/models", (req, res) => {
  const models = [];

  if (gemini) {
    models.push({
      id: `gemini/${GEMINI_MODEL}`,
      object: "model",
      provider: "gemini"
    });
  }

  if (groq) {
    models.push({
      id: `groq/${GROQ_MODEL}`,
      object: "model",
      provider: "groq"
    });
  }

  return res.json({
    object: "list",
    data: models
  });
});

// Chat completions
app.post(
  "/v1/chat/completions",
  rateLimit,
  authenticate,
  async (req, res) => {
    try {
      const {
        messages,
        model,
        temperature = 0.7,
        stream = false
      } = req.body || {};

      if (stream === true) {
        return res.status(400).json({
          error: {
            message: "Streaming is not supported yet",
            type: "invalid_request_error"
          }
        });
      }

      if (
        typeof temperature !== "number" ||
        !Number.isFinite(temperature) ||
        temperature < 0 ||
        temperature > 2
      ) {
        return res.status(400).json({
          error: {
            message: "Temperature must be between 0 and 2",
            type: "invalid_request_error"
          }
        });
      }

      const cleanMessages = normalizeMessages(messages);

      let provider;
      let modelName;

      const explicitModel =
        typeof model === "string" && model.length > 0;

      // Explicit model selection
      if (explicitModel && model.startsWith("gemini/")) {
        provider = "gemini";
        modelName = model.slice("gemini/".length);
      } else if (
        explicitModel &&
        model.startsWith("groq/")
      ) {
        provider = "groq";
        modelName = model.slice("groq/".length);
      } else if (model) {
        return res.status(400).json({
          error: {
            message: "Use gemini/ or groq/ model identifiers",
            type: "invalid_request_error"
          }
        });
      } else if (groq) {
        provider = "groq";
        modelName = GROQ_MODEL;
      } else if (gemini) {
        provider = "gemini";
        modelName = GEMINI_MODEL;
      } else {
        return res.status(503).json({
          error: {
            message: "No AI provider is configured",
            type: "configuration_error"
          }
        });
      }

      // Only configured models are allowed.
      if (
        (provider === "gemini" && modelName !== GEMINI_MODEL) ||
        (provider === "groq" && modelName !== GROQ_MODEL)
      ) {
        return res.status(400).json({
          error: {
            message: "Requested model is not configured",
            type: "invalid_request_error"
          }
        });
      }

      if (provider === "gemini" && !gemini) {
        return res.status(503).json({
          error: {
            message: "Gemini provider is unavailable",
            type: "configuration_error"
          }
        });
      }

      if (provider === "groq" && !groq) {
        return res.status(503).json({
          error: {
            message: "Groq provider is unavailable",
            type: "configuration_error"
          }
        });
      }

      let answer;

      try {
        answer = provider === "gemini"
          ? await askGemini(
              cleanMessages,
              modelName,
              temperature
            )
          : await askGroq(
              cleanMessages,
              modelName,
              temperature
            );
      } catch (primaryError) {
        console.error(
          "Primary provider failed:",
          primaryError.message
        );

        // Do not silently switch providers when the caller
        // explicitly selected a model.
        if (explicitModel) {
          throw primaryError;
        }

        // Automatic fallback is available only when no model
        // was explicitly requested.
        if (provider === "groq" && gemini) {
          provider = "gemini";
          modelName = GEMINI_MODEL;

          answer = await askGemini(
            cleanMessages,
            modelName,
            temperature
          );
        } else if (provider === "gemini" && groq) {
          provider = "groq";
          modelName = GROQ_MODEL;

          answer = await askGroq(
            cleanMessages,
            modelName,
            temperature
          );
        } else {
          throw primaryError;
        }
      }

      if (typeof answer !== "string" || !answer.trim()) {
        throw new Error("Empty AI response");
      }

      return res.json({
        id: `omni-${randomUUID()}`,
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: `${provider}/${modelName}`,
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: answer
            },
            finish_reason: "stop"
          }
        ]
      });
    } catch (error) {
      if (error instanceof ValidationError) {
        return res.status(400).json({
          error: {
            message: error.message,
            type: "invalid_request_error"
          }
        });
      }

      console.error(
        "OMNIVEX request failed:",
        error.message
      );

      return res.status(502).json({
        error: {
          message: "The AI provider could not complete the request",
          type: "provider_error"
        }
      });
    }
  }
);

// Central error handler
app.use((error, req, res, next) => {
  console.error("Request error:", error.message);

  if (res.headersSent) {
    return next(error);
  }

  let status = 400;
  let message = "Invalid request";

  if (error.type === "entity.too.large") {
    status = 413;
    message = "Request body is too large";
  } else if (error.message === "Origin not allowed") {
    status = 403;
    message = "Origin not allowed";
  } else if (error.type === "entity.parse.failed") {
    status = 400;
    message = "Invalid JSON request body";
  }

  return res.status(status).json({
    error: {
      message,
      type: "request_error"
    }
  });
});
// Start server
export default httpServerHandler({ port: PORT });