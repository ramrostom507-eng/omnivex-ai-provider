
import express from "express";
import cors from "cors";
import helmet from "helmet";
import dotenv from "dotenv";
import { GoogleGenerativeAI } from "@google/generative-ai";
import Groq from "groq-sdk";
import { randomUUID, timingSafeEqual } from "node:crypto";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

app.disable("x-powered-by");
app.use(helmet());

const allowedOrigins = (process.env.CORS_ORIGIN || "")
  .split(",")
  .map(x => x.trim().replace(/\/$/, ""))
  .filter(Boolean);

app.use(cors({
  origin(origin, callback) {
    if (!origin || allowedOrigins.includes(origin.replace(/\/$/, ""))) {
      return callback(null, true);
    }
    return callback(new Error("Origin not allowed"));
  }
}));

app.use(express.json({ limit: "1mb" }));

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

// Basic in-memory rate limiter
const requests = new Map();
const RATE_LIMIT = 60;
const WINDOW_MS = 60 * 1000;

function rateLimit(req, res, next) {
  const now = Date.now();
  const ip = req.ip || req.socket.remoteAddress;
  let record = requests.get(ip);

  if (!record || now - record.start > WINDOW_MS) {
    record = { start: now, count: 0 };
  }

  record.count++;
  requests.set(ip, record);

  if (record.count > RATE_LIMIT) {
    return res.status(429).json({
      error: { message: "Too many requests. Try again later." }
    });
  }

  if (requests.size > 10000) {
    for (const [key, value] of requests) {
      if (now - value.start > WINDOW_MS) requests.delete(key);
    }
  }

  next();
}

function authenticate(req, res, next) {
  const apiKey = process.env.OMNIVEX_API_KEY;

  if (!apiKey) {
    return res.status(503).json({
      error: { message: "API authentication is not configured" }
    });
  }

  const authorization = req.headers.authorization || "";
  const suppliedKey = authorization.startsWith("Bearer ")
    ? authorization.slice(7)
    : "";

  const expected = Buffer.from(apiKey);
  const supplied = Buffer.from(suppliedKey);

  if (
    expected.length !== supplied.length ||
    !timingSafeEqual(expected, supplied)
  ) {
    return res.status(401).json({
      error: {
        message: "Invalid API key",
        type: "authentication_error"
      }
    });
  }

  next();
}

function normalizeMessages(messages) {
  if (
    !Array.isArray(messages) ||
    messages.length === 0 ||
    messages.length > 100
  ) {
    throw new Error("Messages must contain between 1 and 100 entries");
  }

  let totalLength = 0;

  return messages.map(message => {
    if (
      !message ||
      !["system", "user", "assistant"].includes(message.role)
    ) {
      throw new Error("Invalid message role");
    }

    const content = typeof message.content === "string"
      ? message.content
      : JSON.stringify(message.content);

    if (!content || content.length > 50000) {
      throw new Error("Invalid message content");
    }

    totalLength += content.length;

    return {
      role: message.role,
      content
    };
  }).filter(message => {
    return message.content.trim().length > 0;
  });
}

async function askGemini(messages, modelName, temperature) {
  if (!gemini) {
    throw new Error("Gemini provider is not configured");
  }

  const systemMessages = messages
    .filter(m => m.role === "system")
    .map(m => m.content);

  const conversation = messages
    .filter(m => m.role !== "system")
    .map(m => {
      const label = m.role === "assistant" ? "Assistant" : "User";
      return `${label}: ${m.content}`;
    })
    .join("\n\n");

  const model = gemini.getGenerativeModel({
    model: modelName,
    systemInstruction: systemMessages.join("\n\n") || undefined,
    generationConfig: { temperature }
  });

  const result = await model.generateContent(conversation);
  return result.response.text();
}

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

  if (!answer) {
    throw new Error("Empty response from Groq");
  }

  return answer;
}

// Main information endpoint
app.get("/", (req, res) => {
  res.json({
    name: "OMNIVEX AI",
    description: "Independent Arabic-first AI provider",
    version: "1.1.0",
    status: "running"
  });
});

// Health check
app.get("/health", (req, res) => {
  res.json({
    status: "healthy",
    service: "OMNIVEX AI Provider",
    version: "1.1.0",
    providers: {
      gemini: Boolean(gemini),
      groq: Boolean(groq)
    }
  });
});

// Available models
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

  res.json({
    object: "list",
    data: models
  });
});

// Chat completions API
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

      if (stream) {
        return res.status(400).json({
          error: { message: "Streaming is not supported yet" }
        });
      }

      if (
        typeof temperature !== "number" ||
        !Number.isFinite(temperature) ||
        temperature < 0 ||
        temperature > 2
      ) {
        return res.status(400).json({
          error: { message: "Temperature must be between 0 and 2" }
        });
      }

      const cleanMessages = normalizeMessages(messages);

      const totalLength = cleanMessages.reduce(
        (sum, message) => sum + message.content.length,
        0
      );

      if (totalLength > 100000) {
        return res.status(413).json({
          error: { message: "Total message size is too large" }
        });
      }

      let provider;
      let modelName;

      if (typeof model === "string" && model.startsWith("gemini/")) {
        provider = "gemini";
        modelName = model.slice(7);
      } else if (typeof model === "string" && model.startsWith("groq/")) {
        provider = "groq";
        modelName = model.slice(5);
      } else if (model) {
        return res.status(400).json({
          error: { message: "Use gemini/ or groq/ model identifiers" }
        });
      } else if (groq) {
        provider = "groq";
        modelName = GROQ_MODEL;
      } else {
        provider = "gemini";
        modelName = GEMINI_MODEL;
      }

      // Only allow configured models
      if (
        (provider === "gemini" && modelName !== GEMINI_MODEL) ||
        (provider === "groq" && modelName !== GROQ_MODEL)
      ) {
        return res.status(400).json({
          error: { message: "Requested model is not configured" }
        });
      }

      if (provider === "gemini" && !gemini) {
        return res.status(503).json({
          error: { message: "Gemini provider is unavailable" }
        });
      }

      if (provider === "groq" && !groq) {
        return res.status(503).json({
          error: { message: "Groq provider is unavailable" }
        });
      }

      let answer;

      try {
        answer = provider === "gemini"
          ? await askGemini(cleanMessages, modelName, temperature)
          : await askGroq(cleanMessages, modelName, temperature);
      } catch (primaryError) {
        console.error("Primary provider failed:", primaryError.message);

        // Automatic fallback when no model was explicitly requested
        if (model || (!gemini && !groq)) {
          throw primaryError;
        }

        if (provider === "groq" && gemini) {
          provider = "gemini";
          modelName = GEMINI_MODEL;
          answer = await askGemini(cleanMessages, modelName, temperature);
        } else if (provider === "gemini" && groq) {
          provider = "groq";
          modelName = GROQ_MODEL;
          answer = await askGroq(cleanMessages, modelName, temperature);
        } else {
          throw primaryError;
        }
      }

      if (!answer || !answer.trim()) {
        throw new Error("Empty AI response");
      }

      return res.json({
        id: `omni-${randomUUID()}`,
        object: "chat.completion",
        created: Math.floor(Date.now() / 1000),
        model: `${provider}/${modelName}`,
        choices: [{
          index: 0,
          message: {
            role: "assistant",
            content: answer
          },
          finish_reason: "stop"
        }]
      });

    } catch (error) {
      console.error("OMNIVEX request failed:", error.message);

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

  if (res.headersSent) return next(error);

  const status = error.type === "entity.too.large" ? 413 : 400;

  res.status(status).json({
    error: {
      message: status === 413
        ? "Request body is too large"
        : "Invalid request or origin"
    }
  });
});

app.listen(PORT, () => {
  console.log(`OMNIVEX AI Provider running on port ${PORT}`);
});
