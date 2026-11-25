import { createContext, useContext, useEffect, useMemo, useState } from "react";
import type { FormEvent, ReactNode } from "react";
import { Link, NavLink, Navigate, Outlet, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import Papa from "papaparse";
import {
  Activity, ArrowDownToLine, ArrowRight, ArrowUpRight, BarChart3, Check, CheckCircle2,
  ChevronDown, CircleHelp, Clock3, Database, FileSpreadsheet, Gauge, Home,
  Layers3, LoaderCircle, LockKeyhole, LogOut, Menu, Network,
  RefreshCw, Search, ShieldCheck, Sparkles, UploadCloud, UserCheck, X,
} from "lucide-react";
import {
  Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api } from "./api";
import type { Model, Prediction, User } from "./api";

type AuthValue = {
  user: User | null;
  loading: boolean;
  signIn: (username: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
  refreshUser: () => Promise<void>;
};
const AuthContext = createContext<AuthValue | null>(null);
const useAuth = () => {
  const value = useContext(AuthContext);
  if (!value) throw new Error("AuthProvider is missing");
  return value;
};

function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const refreshUser = async () => {
    try { setUser(await api.me()); } catch { setUser(null); }
  };
  useEffect(() => { refreshUser().finally(() => setLoading(false)); }, []);
  const value = useMemo<AuthValue>(() => ({
    user, loading,
    signIn: async (username, password) => setUser(await api.login(username, password)),
    signOut: async () => { await api.logout(); setUser(null); },
    refreshUser,
  }), [user, loading]);
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

function App() {
  return <AuthProvider><Routes>
    <Route path="/login" element={<LoginPage />} />
    <Route path="/register" element={<RegisterPage />} />
    <Route element={<ProtectedRoute />}><Route element={<Shell />}>
      <Route index element={<Navigate to="/dashboard" replace />} />
      <Route path="/dashboard" element={<DashboardPage />} />
      <Route path="/models" element={<ModelsPage />} />
      <Route path="/run" element={<RunModelPage />} />
      <Route path="/admin" element={<AdminRoute><AdminPage /></AdminRoute>} />
    </Route></Route>
    <Route path="*" element={<Navigate to="/dashboard" replace />} />
  </Routes></AuthProvider>;
}

function ProtectedRoute() {
  const { user, loading } = useAuth();
  const location = useLocation();
  if (loading) return <LoadingScreen />;
  return user ? <Outlet /> : <Navigate to="/login" state={{ from: location.pathname }} replace />;
}

function LoadingScreen() {
  return <div className="loading-screen"><div className="brand-mark"><Network size={22} /></div><LoaderCircle className="spin" size={22} /><span>Securing your workspace…</span></div>;
}

function AdminRoute({ children }: { children: ReactNode }) {
  const { user, loading } = useAuth();
  if (loading) return <LoadingScreen />;
  return user?.role === "admin" ? <>{children}</> : <Navigate to="/dashboard" replace />;
}

function Shell() {
  const { user, signOut } = useAuth();
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const title = ({ "/dashboard": "Overview", "/models": "Model library", "/run": "Run inference", "/admin": "Access control" } as Record<string, string>)[location.pathname] || "Workspace";
  const logout = async () => { try { await signOut(); navigate("/login"); } catch (e) { toast((e as Error).message, "error"); } };
  return <div className="app-shell">
    <aside className={`sidebar ${mobileOpen ? "sidebar-open" : ""}`}>
      <Link to="/dashboard" className="brand"><span className="brand-mark"><Network size={21} /></span><span>dual<span className="brand-light">path</span><small>MODEL INTELLIGENCE</small></span></Link>
      <div className="workspace-switch"><span className="workspace-icon">DP</span><span><b>Enterprise workspace</b><small>Private environment</small></span><ChevronDown size={14} /></div>
      <div className="nav-label">WORKSPACE</div>
      <nav className="nav-list">
        <NavItem to="/dashboard" icon={<Home size={17} />} label="Overview" />
        <NavItem to="/models" icon={<Layers3 size={17} />} label="Model library" />
        <NavItem to="/run" icon={<Activity size={17} />} label="Run inference" />
      </nav>
      {user?.role === "admin" && <><div className="nav-label nav-label-gap">ADMINISTRATION</div><nav className="nav-list"><NavItem to="/admin" icon={<ShieldCheck size={17} />} label="Access control" /></nav></>}
      <div className="sidebar-bottom">
        <div className="secure-note"><span className="secure-dot" /><div><b>Private by design</b><small>Session protected · Offline-ready</small></div></div>
        <div className="profile-row"><div className="avatar">{user?.full_name?.slice(0, 1).toUpperCase()}</div><div className="profile-meta"><b>{user?.full_name}</b><small>{user?.role === "admin" ? "Administrator" : "Workspace member"}</small></div><button className="icon-button" title="Sign out" onClick={logout}><LogOut size={16} /></button></div>
      </div>
    </aside>
    {mobileOpen && <button className="mobile-scrim" onClick={() => setMobileOpen(false)} aria-label="Close navigation" />}
    <main className="main-area">
      <header className="topbar"><div className="topbar-left"><button className="mobile-menu icon-button" onClick={() => setMobileOpen(!mobileOpen)}><Menu size={19} /></button><span className="crumb">Workspace</span><span className="crumb-sep">/</span><b>{title}</b></div><div className="topbar-right"><span className="live-indicator"><i /> SYSTEM ONLINE</span><div className="topbar-divider" /><span className="topbar-date">{new Date().toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" })}</span><div className="avatar avatar-small">{user?.full_name?.slice(0, 1).toUpperCase()}</div></div></header>
      <div className="page-content"><Outlet /></div>
      <footer className="footer"><span>DUALPATH <i>·</i> PRIVATE MODEL INTELLIGENCE</span><span>Secure workspace <LockKeyhole size={11} /></span></footer>
    </main>
  </div>;
}

function NavItem({ to, icon, label }: { to: string; icon: ReactNode; label: string }) {
  return <NavLink to={to} className={({ isActive }) => `nav-item ${isActive ? "nav-active" : ""}`}>{icon}<span>{label}</span>{label === "Run inference" && <span className="nav-shortcut">↵</span>}</NavLink>;
}

function LoginPage() {
  const { signIn, user } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  if (user) return <Navigate to="/dashboard" replace />;
  const submit = async (e: FormEvent) => {
    e.preventDefault(); setBusy(true); setError("");
    try { await signIn(username, password); navigate((location.state as { from?: string })?.from || "/dashboard", { replace: true }); }
    catch (err) { setError((err as Error).message); } finally { setBusy(false); }
  };
  return <AuthLayout eyebrow="WELCOME BACK" title={<>Your models.<br /><em>Your control.</em></>} subtitle="Sign in to your private model intelligence workspace.">
    <form className="auth-form" onSubmit={submit}><div className="field"><label>USERNAME OR EMAIL</label><input autoComplete="username" value={username} onChange={e => setUsername(e.target.value)} placeholder="Enter your username or email" required /></div><div className="field"><label>PASSWORD</label><input type="password" autoComplete="current-password" value={password} onChange={e => setPassword(e.target.value)} placeholder="Enter your password" required /></div>
      {error && <div className="form-error"><X size={15} />{error}</div>}<button className="primary-button full-button" disabled={busy}>{busy ? <LoaderCircle className="spin" size={17} /> : <>Enter workspace <ArrowRight size={17} /></>}</button>
    </form><p className="auth-switch">New to DualPath? <Link to="/register">Request access <ArrowUpRight size={13} /></Link></p>
  </AuthLayout>;
}

function RegisterPage() {
  const [form, setForm] = useState({ username: "", email: "", full_name: "", password: "" });
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState(false);
  const [error, setError] = useState("");
  const update = (key: keyof typeof form, value: string) => setForm(old => ({ ...old, [key]: value }));
  const submit = async (e: FormEvent) => {
    e.preventDefault(); setBusy(true); setError("");
    try { await api.register(form); setDone(true); } catch (err) { setError((err as Error).message); } finally { setBusy(false); }
  };
  return <AuthLayout eyebrow="REQUEST ACCESS" title={<>A secure space<br /><em>to explore.</em></>} subtitle="Create your account. An administrator will review your request.">
    {done ? <div className="success-panel"><span className="success-icon"><CheckCircle2 size={25} /></span><h3>Request received</h3><p>Your account is pending administrator approval. You can sign in once access is granted.</p><Link className="primary-button full-button" to="/login">Back to sign in <ArrowRight size={16} /></Link></div> :
    <form className="auth-form" onSubmit={submit}><div className="field"><label>FULL NAME</label><input value={form.full_name} onChange={e => update("full_name", e.target.value)} placeholder="Your full name" required minLength={2} /></div><div className="field"><label>USERNAME</label><input value={form.username} onChange={e => update("username", e.target.value)} placeholder="Choose a username" required minLength={3} /></div><div className="field"><label>WORK EMAIL</label><input type="email" value={form.email} onChange={e => update("email", e.target.value)} placeholder="you@company.com" required /></div><div className="field"><label>PASSWORD <span className="field-hint">12 characters minimum</span></label><input type="password" value={form.password} onChange={e => update("password", e.target.value)} placeholder="Create a strong password" required minLength={12} /></div>
      {error && <div className="form-error"><X size={15} />{error}</div>}<button className="primary-button full-button" disabled={busy}>{busy ? <LoaderCircle className="spin" size={17} /> : <>Submit access request <ArrowRight size={17} /></>}</button></form>}
    <p className="auth-switch">Already have access? <Link to="/login">Sign in <ArrowUpRight size={13} /></Link></p>
  </AuthLayout>;
}

function AuthLayout({ eyebrow, title, subtitle, children }: { eyebrow: string; title: ReactNode; subtitle: string; children: ReactNode }) {
  return <div className="auth-page"><div className="auth-visual"><div className="auth-visual-grid" /><Link to="/login" className="brand auth-brand"><span className="brand-mark"><Network size={21} /></span><span>dual<span className="brand-light">path</span><small>MODEL INTELLIGENCE</small></span></Link><div className="auth-hero-copy"><div className="eyebrow"><span /> PRIVATE MODEL ENVIRONMENT</div><h1>Validation,<br />without <em>compromise.</em></h1><p>One secure workspace for model evaluation, inference and intelligent decision support.</p><div className="hero-orbit"><div className="orbit orbit-one" /><div className="orbit orbit-two" /><div className="orbit-core"><Network size={33} /></div><span className="orbit-node node-a"><ShieldCheck size={17} /></span><span className="orbit-node node-b"><Database size={17} /></span><span className="orbit-node node-c"><Activity size={17} /></span></div></div><div className="auth-visual-foot"><span>SECURE BY ARCHITECTURE</span><span>01 / 04</span></div></div><div className="auth-panel"><div className="auth-panel-inner"><div className="auth-eyebrow">{eyebrow}</div><h2>{title}</h2><p className="auth-subtitle">{subtitle}</p>{children}<div className="auth-foot"><LockKeyhole size={13} /> Protected session · Administrator-approved access</div></div></div></div>;
}

function DashboardPage() {
  const [models, setModels] = useState<Model[]>([]);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState("");
  const { user } = useAuth();
  const load = async () => { setBusy(true); try { setModels(await api.models()); setError(""); } catch (e) { setError((e as Error).message); } finally { setBusy(false); } };
  useEffect(() => { load(); }, []);
  const ready = models.filter(m => m.is_available).length;
  const families = [...new Set(models.map(m => m.family))].length;
  return <div className="page-enter"><div className="welcome-row"><div><div className="eyebrow"><span /> YOUR PRIVATE WORKSPACE</div><h1>Good to see you, {user?.full_name?.split(" ")[0]}<span className="heading-period">.</span></h1><p className="page-lead">Your model environment is ready. Here's the current workspace snapshot.</p></div><Link to="/run" className="primary-button"><Sparkles size={16} /> Start inference <ArrowRight size={16} /></Link></div>
    <div className="stats-grid"><StatCard icon={<Layers3 />} label="REGISTERED MODELS" value={busy ? "—" : String(models.length)} detail="Across all model families" /><StatCard icon={<CheckCircle2 />} label="READY FOR INFERENCE" value={busy ? "—" : String(ready)} detail={models.length ? `${Math.round(ready / models.length * 100)}% of catalog available` : "Waiting for catalog"} tone="green" /><StatCard icon={<Network />} label="MODEL FAMILIES" value={busy ? "—" : String(families)} detail="Distinct architecture groups" /><StatCard icon={<ShieldCheck />} label="ACCESS STATUS" value="Verified" detail="Administrator-approved session" tone="violet" /></div>
    {error && <div className="inline-alert"><X size={16} />{error}<button onClick={load}>Retry</button></div>}
    <div className="section-heading"><div><div className="eyebrow">EXPLORE YOUR TOOLKIT</div><h2>Model landscape</h2></div><Link to="/models" className="text-link">View all models <ArrowRight size={15} /></Link></div>
    <div className="model-preview-grid">{busy ? <div className="loading-card"><LoaderCircle className="spin" /> Loading model catalog…</div> : models.slice(0, 4).map((m, i) => <ModelCard key={m.key} model={m} index={i} />)}</div>
    <div className="bottom-callout"><div className="callout-icon"><FileSpreadsheet size={22} /></div><div><b>Have a benchmark dataset ready?</b><p>Upload a CSV and evaluate it against one of your available models.</p></div><Link to="/run" className="secondary-button">Upload dataset <ArrowUpRight size={15} /></Link></div>
  </div>;
}

function StatCard({ icon, label, value, detail, tone = "blue" }: { icon: ReactNode; label: string; value: string; detail: string; tone?: string }) {
  return <div className="stat-card"><div className={`stat-icon ${tone}`}>{icon}</div><div className="stat-label">{label}</div><div className="stat-value">{value}</div><div className="stat-detail">{detail}</div></div>;
}

function ModelCard({ model, index }: { model: Model; index: number }) {
  const icon = model.family.includes("keras") || model.family.includes("dl") ? <Activity /> : model.family.includes("embedding") || model.family.includes("transformer") ? <Network /> : <Gauge />;
  return <div className="model-card" style={{ animationDelay: `${index * 70}ms` }}><div className="model-card-top"><span className="model-icon">{icon}</span><span className={`status-pill ${model.is_available ? "status-ready" : "status-muted"}`}><i />{model.is_available ? "READY" : "UNAVAILABLE"}</span></div><div className="model-name-row"><h3>{model.display_name}</h3>{model.is_proposed && <span className="proposed-tag">PROPOSED</span>}</div><p>{model.description}</p><div className="model-card-foot"><span>{model.family.replaceAll("_", " ")}</span><span className="target-tag">{model.target_column}</span></div>{!model.is_available && model.unavailable_reason && <div className="unavailable-reason" title={model.unavailable_reason}>Requires setup</div>}</div>;
}

function ModelsPage() {
  const [models, setModels] = useState<Model[]>([]);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("all");
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState("");
  const load = async () => { setBusy(true); try { setModels(await api.models()); setError(""); } catch (e) { setError((e as Error).message); } finally { setBusy(false); } };
  useEffect(() => { load(); }, []);
  const families = ["all", ...new Set(models.map(m => m.family))];
  const visible = models.filter(m => (filter === "all" || m.family === filter) && `${m.display_name} ${m.description} ${m.key}`.toLowerCase().includes(query.toLowerCase()));
  return <div className="page-enter"><PageIntro eyebrow="MODEL CATALOG" title="The model library." subtitle="Browse the registered architectures and check which models are ready to run." action={<button className="secondary-button" onClick={load}><RefreshCw size={15} /> Refresh catalog</button>} />
    <div className="catalog-toolbar"><div className="search-box"><Search size={16} /><input value={query} onChange={e => setQuery(e.target.value)} placeholder="Search models…" /></div><div className="filter-pills">{families.map(f => <button key={f} className={filter === f ? "filter-active" : ""} onClick={() => setFilter(f)}>{f === "all" ? "All models" : f.replaceAll("_", " ")}</button>)}</div></div>
    {error && <div className="inline-alert">{error}</div>}{busy ? <div className="loading-card"><LoaderCircle className="spin" /> Loading models…</div> : <><div className="catalog-summary"><span><b>{visible.length}</b> models shown</span><span><i className="legend-dot green-dot" /> {models.filter(m => m.is_available).length} ready <i className="legend-dot grey-dot" /> {models.filter(m => !m.is_available).length} unavailable</span></div><div className="models-grid">{visible.map((m, i) => <ModelCard key={m.key} model={m} index={i} />)}</div>{visible.length === 0 && <div className="empty-state"><Search size={25} /><b>No matching models</b><span>Try another search or category.</span></div>}</>}
    <div className="notice-panel"><CircleHelp size={18} /><div><b>About model readiness</b><p>Readiness is determined by the backend from the presence of trusted artifacts and a registered inference adapter. Models without a complete, verified bundle remain unavailable.</p></div></div>
  </div>;
}

function PageIntro({ eyebrow, title, subtitle, action }: { eyebrow: string; title: string; subtitle: string; action?: ReactNode }) {
  return <div className="page-intro"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p className="page-lead">{subtitle}</p></div>{action}</div>;
}

function RunModelPage() {
  const [models, setModels] = useState<Model[]>([]);
  const [modelKey, setModelKey] = useState("");
  const [records, setRecords] = useState<Record<string, unknown>[]>([]);
  const [fileName, setFileName] = useState("");
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const [stage, setStage] = useState<"upload" | "results">("upload");
  useEffect(() => { api.models().then(data => { const ready = data.filter(m => m.is_available); setModels(ready); if (ready[0]) setModelKey(ready[0].key); }).catch(e => setError(e.message)).finally(() => setLoading(false)); }, []);
  const selected = models.find(m => m.key === modelKey);
  const onFile = (file?: File) => {
    if (!file) return;
    setError(""); setPredictions([]); setStage("upload");
    if (!file.name.toLowerCase().endsWith(".csv")) { setError("Please upload a CSV file. Excel files are not supported in this version."); return; }
    Papa.parse<Record<string, unknown>>(file, {
      header: true, skipEmptyLines: "greedy", dynamicTyping: true,
      complete: result => {
        if (result.errors.length) { setError(result.errors[0].message); return; }
        const rows = result.data.filter(row => Object.values(row).some(value => value !== "" && value !== null && value !== undefined));
        if (!rows.length) { setError("The CSV contains no data rows."); return; }
        if (rows.length > 1000) { setError("The API currently accepts up to 1,000 records per request. Split your CSV into smaller files."); return; }
        setRecords(rows); setFileName(file.name);
      },
      error: err => setError(err.message),
    });
  };
  const run = async () => {
    if (!selected || !records.length) return;
    setBusy(true); setError("");
    try { const result = await api.predict(modelKey, records); setPredictions(result.predictions); setStage("results"); toast(`Inference completed for ${result.predictions.length} records.`, "success"); }
    catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  };
  const reset = () => { setRecords([]); setPredictions([]); setFileName(""); setStage("upload"); setError(""); };
  const download = () => {
    const output = records.map((record, i) => ({ ...record, predicted_label: predictions[i]?.predicted_label, confidence: predictions[i]?.confidence, ...Object.fromEntries(Object.entries(predictions[i]?.class_probabilities || {}).map(([key, value]) => [`probability_${key}`, value])) }));
    const csv = Papa.unparse(output); const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    const a = document.createElement("a"); a.href = url; a.download = `dualpath-${modelKey}-results.csv`; a.click(); URL.revokeObjectURL(url);
  };
  const distribution = Object.entries(predictions.reduce<Record<string, number>>((acc, p) => { const key = String(p.predicted_label); acc[key] = (acc[key] || 0) + 1; return acc; }, {})).map(([name, value]) => ({ name, value }));
  const averageConfidence = predictions.length ? predictions.reduce((sum, p) => sum + p.confidence, 0) / predictions.length : 0;
  return <div className="page-enter"><PageIntro eyebrow="INFERENCE WORKBENCH" title="Run an evaluation." subtitle="Choose a ready model, upload a benchmark CSV and inspect the predictions." />
    {error && <div className="inline-alert"><X size={16} />{error}<button onClick={() => setError("")}>Dismiss</button></div>}
    {stage === "upload" ? <div className="run-layout"><div className="run-main"><div className="panel-card"><div className="panel-heading"><div><span className="step-index">01</span><div><h3>Select a model</h3><p>Only models with complete inference bundles are selectable.</p></div></div><span className="step-state"><CheckCircle2 size={14} /> STEP ONE</span></div>
      {loading ? <div className="loading-card"><LoaderCircle className="spin" /> Loading available models…</div> : models.length ? <div className="select-wrap"><select value={modelKey} onChange={e => setModelKey(e.target.value)}>{models.map(m => <option key={m.key} value={m.key}>{m.display_name} · {m.family}</option>)}</select><ChevronDown size={16} /></div> : <div className="empty-inline">No models are currently available. Check the model catalog and backend artifact configuration.</div>}
      {selected && <div className="selected-model"><div className="model-icon"><Activity /></div><div><b>{selected.display_name}</b><span>{selected.description}</span></div><span className="status-pill status-ready"><i /> READY</span></div>}
    </div>
    <div className="panel-card"><div className="panel-heading"><div><span className="step-index">02</span><div><h3>Upload benchmark data</h3><p>Import a CSV file containing the model's required feature columns.</p></div></div><span className="step-state">STEP TWO</span></div>
      <label className={`drop-zone ${dragging ? "drop-active" : ""} ${records.length ? "drop-loaded" : ""}`} onDragOver={e => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={e => { e.preventDefault(); setDragging(false); onFile(e.dataTransfer.files[0]); }}>
        <input type="file" accept=".csv,text/csv" onChange={e => onFile(e.target.files?.[0])} />
        {records.length ? <><span className="upload-symbol upload-success"><CheckCircle2 size={25} /></span><b>{fileName}</b><span>{records.length.toLocaleString()} records parsed successfully</span><small>Choose another file to replace this dataset</small></> : <><span className="upload-symbol"><UploadCloud size={25} /></span><b>Drop your CSV file here</b><span>or <u>browse files</u> from your computer</span><small>CSV only · Maximum 1,000 records per request</small></>}
      </label>
      {records.length > 0 && <div className="dataset-meta"><span><FileSpreadsheet size={15} /> {Object.keys(records[0]).length} columns</span><span><Database size={15} /> {records.length.toLocaleString()} rows</span><button onClick={reset}><X size={14} /> Remove</button></div>}
      {records.length > 0 && <div className="table-wrap preview-table"><table><thead><tr>{Object.keys(records[0]).slice(0, 5).map(key => <th key={key}>{key}</th>)}</tr></thead><tbody>{records.slice(0, 4).map((row, i) => <tr key={i}>{Object.keys(records[0]).slice(0, 5).map(key => <td key={key}>{String(row[key] ?? "—").slice(0, 48)}</td>)}</tr>)}</tbody></table><div className="table-foot">Showing first {Math.min(4, records.length)} of {records.length} records · {Object.keys(records[0]).length > 5 ? `${Object.keys(records[0]).length - 5} more columns` : "All columns shown"}</div></div>}
    </div>
    <div className="run-actions"><div><ShieldCheck size={15} /><span>Data is sent to your configured private API.</span></div><button className="primary-button" disabled={!selected || !records.length || busy} onClick={run}>{busy ? <><LoaderCircle className="spin" size={16} /> Running inference…</> : <>Run evaluation <ArrowRight size={16} /></>}</button></div>
    </div>
    <aside className="run-aside"><div className="aside-card"><div className="aside-card-icon"><CircleHelp size={18} /></div><h3>Before you upload</h3><p>The current TF-IDF/Keras checkpoints are post-execution benchmark classifiers. They need generated SQL and execution metrics, not only a natural-language query.</p><Link to="/models" className="text-link">Review model requirements <ArrowRight size={14} /></Link></div><div className="aside-card aside-muted"><div className="aside-card-icon"><FileSpreadsheet size={18} /></div><h3>Expected feature schema</h3><p>For TF-IDF/Keras: Level, NL_Query, Expected_SQL, Generated_SQL and benchmark metrics including QueryID, row counts, match percentages, confidence, operation/column match and generation time.</p><span className="aside-foot">See docs/TFIDF_KERAS_INFERENCE.md</span></div></aside></div> :
    <div className="results-view"><div className="results-header"><div><span className="result-success"><CheckCircle2 size={15} /> EVALUATION COMPLETE</span><h2>Inference results</h2><p>{fileName} · {selected?.display_name}</p></div><div className="result-actions"><button className="secondary-button" onClick={reset}>New evaluation</button><button className="primary-button" onClick={download}><ArrowDownToLine size={16} /> Export CSV</button></div></div>
      <div className="stats-grid results-stats"><StatCard icon={<Database />} label="RECORDS EVALUATED" value={String(predictions.length)} detail="Predictions returned by API" /><StatCard icon={<Gauge />} label="AVERAGE CONFIDENCE" value={`${(averageConfidence * 100).toFixed(1)}%`} detail="Mean maximum class score" tone="green" /><StatCard icon={<Layers3 />} label="PREDICTED CLASSES" value={String(distribution.length)} detail="Unique output labels" tone="violet" /><StatCard icon={<Clock3 />} label="PREDICTION STAGE" value="Post-execution" detail="Benchmark feature inference" /></div>
      <div className="charts-grid"><div className="chart-card"><div className="chart-heading"><div><h3>Prediction distribution</h3><p>Records by predicted class</p></div><BarChart3 size={18} /></div><div className="chart-area"><ResponsiveContainer width="100%" height="100%"><PieChart><Pie data={distribution} dataKey="value" nameKey="name" innerRadius={62} outerRadius={94} paddingAngle={4}>{distribution.map((_, i) => <Cell key={i} fill={["#78a8ff", "#9b8cff", "#4bd6b0", "#f5b86b", "#f17e9a"][i % 5]} />)}</Pie><Tooltip contentStyle={{ background: "#111a2b", border: "1px solid #28364d", borderRadius: 10, color: "#fff" }} /></PieChart></ResponsiveContainer></div><div className="chart-legend">{distribution.map((d, i) => <span key={d.name}><i style={{ background: ["#78a8ff", "#9b8cff", "#4bd6b0", "#f5b86b", "#f17e9a"][i % 5] }} />{d.name} <b>{d.value}</b></span>)}</div></div>
      <div className="chart-card"><div className="chart-heading"><div><h3>Confidence by record</h3><p>Maximum class score per prediction</p></div><Gauge size={18} /></div><div className="chart-area"><ResponsiveContainer width="100%" height="100%"><BarChart data={predictions.slice(0, 20).map((p, i) => ({ record: String(i + 1), confidence: Number((p.confidence * 100).toFixed(1)) }))} margin={{ top: 10, right: 8, left: -18, bottom: 0 }}><CartesianGrid stroke="#263248" strokeDasharray="3 3" vertical={false} /><XAxis dataKey="record" tick={{ fill: "#8998ae", fontSize: 10 }} axisLine={false} tickLine={false} /><YAxis domain={[0, 100]} tick={{ fill: "#8998ae", fontSize: 10 }} axisLine={false} tickLine={false} /><Tooltip contentStyle={{ background: "#111a2b", border: "1px solid #28364d", borderRadius: 10 }} /><Bar dataKey="confidence" fill="#79a8ff" radius={[5, 5, 0, 0]} /></BarChart></ResponsiveContainer></div><div className="chart-foot">First {Math.min(20, predictions.length)} records shown</div></div></div>
      <div className="panel-card prediction-table-card"><div className="panel-heading"><div><span className="step-index"><Check size={15} /></span><div><h3>Prediction details</h3><p>Class label and model confidence for every record.</p></div></div><span className="step-state">{predictions.length} ROWS</span></div><div className="table-wrap"><table><thead><tr><th>RECORD</th><th>PREDICTED LABEL</th><th>CONFIDENCE</th><th>CLASS PROBABILITIES</th></tr></thead><tbody>{predictions.slice(0, 100).map((p, i) => <tr key={i}><td className="mono">#{String(i + 1).padStart(3, "0")}</td><td><span className="prediction-label">{String(p.predicted_label)}</span></td><td><div className="confidence-cell"><div className="confidence-track"><i style={{ width: `${Math.max(0, Math.min(100, p.confidence * 100))}%` }} /></div><b>{(p.confidence * 100).toFixed(1)}%</b></div></td><td className="probability-cell">{Object.entries(p.class_probabilities).map(([label, value]) => `${label}: ${(value * 100).toFixed(1)}%`).join(" · ")}</td></tr>)}</tbody></table></div>{predictions.length > 100 && <div className="table-foot">Showing first 100 rows. Export the CSV for the complete result set.</div>}</div>
    </div>}
  </div>;
}

function AdminPage() {
  const [users, setUsers] = useState<User[]>([]);
  const [busy, setBusy] = useState(true);
  const [workingId, setWorkingId] = useState<number | null>(null);
  const [error, setError] = useState("");
  const load = async () => { setBusy(true); try { setUsers(await api.pendingUsers()); setError(""); } catch (e) { setError((e as Error).message); } finally { setBusy(false); } };
  useEffect(() => { load(); }, []);
  const act = async (id: number, action: "approve" | "reject") => {
    setWorkingId(id); try { if (action === "approve") await api.approveUser(id); else await api.rejectUser(id); toast(action === "approve" ? "Account approved." : "Account rejected.", action === "approve" ? "success" : "info"); await load(); } catch (e) { setError((e as Error).message); } finally { setWorkingId(null); }
  };
  return <div className="page-enter"><PageIntro eyebrow="ADMINISTRATION" title="Access control." subtitle="Review new account requests and decide who can enter the private workspace." action={<button className="secondary-button" onClick={load}><RefreshCw size={15} /> Refresh</button>} />
    <div className="admin-banner"><div className="admin-banner-icon"><ShieldCheck size={22} /></div><div><b>Administrator console</b><p>Only approved accounts can access model catalogs and inference endpoints. Every decision is recorded by the backend audit log.</p></div><span className="admin-secure">ADMIN ONLY</span></div>
    <div className="section-heading"><div><div className="eyebrow">ACCESS REQUESTS</div><h2>Pending approval <span className="count-bubble">{users.length}</span></h2></div><span className="muted-meta">Oldest requests first</span></div>
    {error && <div className="inline-alert"><X size={16} />{error}</div>}
    {busy ? <div className="loading-card"><LoaderCircle className="spin" /> Loading requests…</div> : users.length ? <div className="requests-list">{users.map(u => <div className="request-row" key={u.id}><div className="avatar request-avatar">{u.full_name.slice(0, 1).toUpperCase()}</div><div className="request-person"><b>{u.full_name}</b><span>@{u.username} · {u.email}</span></div><div className="request-date"><Clock3 size={13} /> {new Date(u.created_at).toLocaleDateString()}</div><span className="status-pill status-pending"><i /> PENDING</span><div className="request-actions"><button className="approve-button" disabled={workingId === u.id} onClick={() => act(u.id, "approve")}>{workingId === u.id ? <LoaderCircle className="spin" size={14} /> : <Check size={15} />} Approve</button><button className="reject-button" disabled={workingId === u.id} onClick={() => act(u.id, "reject")}><X size={15} /> Reject</button></div></div>)}</div> : <div className="empty-state"><span className="empty-check"><UserCheck size={25} /></span><b>You're all caught up</b><span>There are no pending account requests right now.</span></div>}
  </div>;
}

function toast(message: string, type: "success" | "error" | "info" = "info") {
  const event = new CustomEvent("dualpath-toast", { detail: { message, type } });
  window.dispatchEvent(event);
}
function ToastHost() {
  const [items, setItems] = useState<{ id: number; message: string; type: string }[]>([]);
  useEffect(() => {
    const handler = (event: Event) => {
      const detail = (event as CustomEvent).detail;
      const id = Date.now() + Math.random();
      setItems(old => [...old, { id, ...detail }]);
      window.setTimeout(() => setItems(old => old.filter(item => item.id !== id)), 3600);
    };
    window.addEventListener("dualpath-toast", handler);
    return () => window.removeEventListener("dualpath-toast", handler);
  }, []);
  return <div className="toast-stack">{items.map(item => <div className={`toast toast-${item.type}`} key={item.id}><CheckCircle2 size={16} />{item.message}<button onClick={() => setItems(old => old.filter(x => x.id !== item.id))}><X size={14} /></button></div>)}</div>;
}

export default function AppRoot() {
  return <><App /><ToastHost /></>;
}
