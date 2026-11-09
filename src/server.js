
import express from "express";
import cors from "cors";
import helmet from "helmet";
import dotenv from "dotenv";
import { GoogleGenerativeAI } from "@google/generative-ai";
import Groq from "groq-sdk";
import { randomUUID } from "node:crypto";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

const allowedOrigins = (process.env.CORS_ORIGIN || "")
  .split(",")
  .map((origin) => origin.trim())
  .filter(Boolean);

app.use(helmet());
app.use(cors({
  origin: allowedOrigins.length ? allowedOrigins : true
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

function authenticate(req, res, next) {
  const apiKey = process.env.OMNIVEX_API_KEY;

  if (!apiKey) return next();

  const authorization = req.headers.authorization || "";
  const suppliedKey = authorization.startsWith("Bearer ")
    ? authorization.slice(7)
    : "";

  if (suppliedKey !== apiKey) {
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
  return messages
    .filter((message) =>
      ["system", "user", "assistant"].includes(message.role)
    )
    .map((message) => ({
      role: message.role,
      content: typeof message.content === "string"
        ? message.content
        : JSON.stringify(message.content)
    }));
}

async function askGemini(messages, modelName, temperature) {
  if (!gemini) {
    throw new Error("Gemini provider is not configured");
  }

  const model = gemini.getGenerativeModel({
    model: modelName,
    generationConfig: { temperature }
  });

  const prompt = messages
    .map((message) => {
      const label = {
        system: "System",
        user: "User",
        assistant: "Assistant"
      }[message.role];

      return `${label}: ${message.content}`;
    })
    .join("\n\n");

  const result = await model.generateContent(prompt);
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

  return result.choices?.[0]?.message?.content || "";
}

app.get("/", (req, res) => {
  res.json({
    name: "OMNIVEX AI",
    description: "Independent Arabic-first AI provider",
    status: "running"
  });
});

app.get("/health", (req, res) => {
  res.json({
    status: "healthy",
    service: "OMNIVEX AI Provider",
    providers: {
      gemini: Boolean(gemini),
      groq: Boolean(groq)
    }
  });
});

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

app.post("/v1/chat/completions", authenticate, async (req, res) => {
  try {
    const { messages, model, temperature = 0.7, stream = false } = req.body;

    if (!Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({
        error: { message: "A non-empty messages array is required" }
      });
    }

    if (stream) {
      return res.status(400).json({
        error: { message: "Streaming is not supported yet" }
      });
    }

    const cleanMessages = normalizeMessages(messages);

    if (!cleanMessages.length) {
      return res.status(400).json({
        error: { message: "No valid messages were supplied" }
      });
    }

    let provider;
    let modelName;

    if (model?.startsWith("gemini/")) {
      provider = "gemini";
      modelName = model.slice("gemini/".length);
    } else if (model?.startsWith("groq/")) {
      provider = "groq";
      modelName = model.slice("groq/".length);
    } else if (model?.startsWith("gemini")) {
      provider = "gemini";
      modelName = model;
    } else if (model) {
      provider = "groq";
      modelName = model;
    } else if (groq) {
      provider = "groq";
      modelName = GROQ_MODEL;
    } else {
      provider = "gemini";
      modelName = GEMINI_MODEL;
    }

    const answer = provider === "gemini"
      ? await askGemini(cleanMessages, modelName, temperature)
      : await askGroq(cleanMessages, modelName, temperature);

    res.json({
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

    res.status(500).json({
      error: {
        message: "The AI provider could not complete the request",
        type: "provider_error"
      }
    });
  }
});

app.listen(PORT, () => {
  console.log(`OMNIVEX AI Provider running on port ${PORT}`);
});
