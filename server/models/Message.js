// server/models/Message.js
const mongoose = require('mongoose');
const MessageSchema = new mongoose.Schema({
  text: { type: String, required: true },
  clean: { type: String },
  label: { type: Number, default: -1 }, // -1 unknown, 0 ham, 1 spam
  prob: { type: Number, default: 0 },
  anomaly: { type: Boolean, default: false },
  createdAt: { type: Date, default: Date.now }
});
module.exports = mongoose.model('Message', MessageSchema);
