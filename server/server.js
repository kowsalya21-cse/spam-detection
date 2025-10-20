// server/server.js
require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const axios = require('axios');

const app = express();

// Middleware
app.use(cors());
app.use(express.json()); // parse JSON request body

// Environment variables
const PORT = process.env.PORT || 5002;
const API_URL = process.env.API_URL || "http://localhost:8001"; // Python ML service URL
const MONGO_URI = process.env.MONGO_URI || "mongodb://localhost:27017/spamdb";

// MongoDB connection
(async () => {
  try {
    await mongoose.connect(MONGO_URI, {
      useNewUrlParser: true,
      useUnifiedTopology: true
    });
    console.log("✅ MongoDB connected");
  } catch (err) {
    console.error("❌ MongoDB connection error:", err);
    process.exit(1);
  }
})();

// Message schema
const messageSchema = new mongoose.Schema({
  text: { type: String, required: true },
  clean: String,
  label: { type: Number, default: 0 },
  prob: { type: Number, default: 0.0 },
  anomaly: { type: Boolean, default: false },
}, { timestamps: true });

const Message = mongoose.model("Message", messageSchema);

// POST /api/messages - classify SMS and save
app.post("/api/messages", async (req, res) => {
  try {
    const { text } = req.body;
    if (!text) return res.status(400).json({ error: "Text is required" });

    // Call Python ML service
    const mlResponse = await axios.post(`${API_URL}/predict`, { text });

    // Save result to MongoDB
    const newMessage = new Message({
      text,
      clean: mlResponse.data.clean,
      label: mlResponse.data.label,
      prob: mlResponse.data.prob,
      anomaly: mlResponse.data.anomaly
    });

    await newMessage.save();
    res.json(newMessage);

  } catch (err) {
    console.error("❌ POST /api/messages error:", err.message || err);
    res.status(500).json({ error: "Something went wrong" });
  }
});

// GET /api/messages - fetch recent messages
app.get("/api/messages", async (req, res) => {
  try {
    const messages = await Message.find().sort({ createdAt: -1 }).lean();

    // Fill missing fields safely
    const safeMessages = messages.map(msg => ({
      _id: msg._id,
      text: msg.text || "",
      clean: msg.clean || "",
      label: msg.label !== undefined ? msg.label : 0,
      prob: msg.prob !== undefined ? msg.prob : 0.0,
      anomaly: msg.anomaly !== undefined ? msg.anomaly : false,
      createdAt: msg.createdAt || new Date()
    }));

    res.json(safeMessages);

  } catch (err) {
    console.error("❌ GET /api/messages error:", err.message || err);
    res.status(500).json({ error: "Something went wrong" });
  }
});

// Root route
app.get("/", (req, res) => {
  res.json({ status: "Spam Detection Server Running" });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Server listening on port ${PORT}`);
});
