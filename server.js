import express from "express";
import { spawn } from "child_process";
import readline from "readline";

const app = express();

app.use(express.json());
app.use(express.static("public"));

const python = spawn("python", ["chat_backend.py"]);

const rl = readline.createInterface({
    input: python.stdout
});

let pendingResponse = null;
let backendReady = false;

rl.on("line", (line) => {
    try {
        const data = JSON.parse(line);

        if (data.status === "ready") {
            backendReady = true;
            console.log("Python backend ready.");
            return;
        }

        if (pendingResponse) {
            pendingResponse.json(data);
            pendingResponse = null;
        }

    } catch (err) {
        console.error("Python Output:", line);
    }
});

python.stderr.on("data", (data) => {
    console.error("Python Error:");
    console.error(data.toString());
});

app.post("/api/chat", (req, res) => {

    if (!backendReady) {
        return res.status(503).json({
            error: "Python backend is still loading."
        });
    }

    const message = req.body.message;

    if (!message) {
        return res.status(400).json({
            error: "Message is required."
        });
    }

    pendingResponse = res;

    python.stdin.write(message + "\n");
});

app.listen(8080, () => {
    console.log("Server running at http://localhost:8080");
});