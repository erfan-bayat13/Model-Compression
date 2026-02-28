import { Route, Routes } from "react-router-dom";

import { Configure } from "./pages/Configure";
import { Home } from "./pages/Home";
import { Progress } from "./pages/Progress";
import { Results } from "./pages/Results";

export function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-[var(--border)] px-6 py-3.5 flex-shrink-0">
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent)]" />
          <span className="text-sm font-medium font-mono text-[var(--text-primary)] tracking-tight">
            model-compression
          </span>
        </div>
      </header>

      <main className="flex-1">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/configure" element={<Configure />} />
          <Route path="/progress/:jobId" element={<Progress />} />
          <Route path="/results/:jobId" element={<Results />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
