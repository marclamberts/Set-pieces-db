import React, { useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  BarChart3,
  Crosshair,
  ChevronLeft,
  Goal,
  Shield,
  Users,
  Upload,
  Filter,
  Map,
  Table2,
  Activity,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ScatterChart,
  Scatter,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
} from "recharts";

type SetPieceType = "Corner" | "Free Kick" | "Throw-In";

type EventRow = {
  match_id: string;
  Match: string;
  team: string;
  Minute: number;
  Second: number;
  Taker: string;
  Shooter: string;
  set_piece_type: SetPieceType;
  shot_xg: number;
  led_to_shot: boolean;
  goal: boolean;
  side: "Left" | "Right" | "Unknown";
  delivery_zone: string;
  end_zone: string;
  phase: string;
  shot_location_x?: number;
  shot_location_y?: number;
  pass_location_x?: number;
  pass_location_y?: number;
  pass_end_location_x?: number;
  pass_end_location_y?: number;
};

const TYPE_META: Record<SetPieceType, { color: string; accent: string; desc: string; icon: React.ReactNode }> = {
  Corner: {
    color: "from-sky-500/20 to-sky-500/5",
    accent: "bg-sky-400",
    desc: "Wide dead-ball delivery, targeting, taker profiles, and shot creation.",
    icon: <Goal className="h-5 w-5" />,
  },
  "Free Kick": {
    color: "from-emerald-500/20 to-emerald-500/5",
    accent: "bg-emerald-400",
    desc: "Direct and indirect routines, delivery quality, and final-third threat.",
    icon: <Crosshair className="h-5 w-5" />,
  },
  "Throw-In": {
    color: "from-orange-500/20 to-orange-500/5",
    accent: "bg-orange-400",
    desc: "Attacking throw-ins, long-throw patterns, zones, and team usage.",
    icon: <Shield className="h-5 w-5" />,
  },
};

const PIE_COLORS = ["#38bdf8", "#34d399", "#fb923c", "#a78bfa", "#fbbf24", "#fb7185"];

const demoData: EventRow[] = [
  {
    match_id: "1",
    Match: "Malmö FF - AIK",
    team: "Malmö FF",
    Minute: 12,
    Second: 14,
    Taker: "Player A",
    Shooter: "Player B",
    set_piece_type: "Corner",
    shot_xg: 0.11,
    led_to_shot: true,
    goal: false,
    side: "Left",
    delivery_zone: "Central Zone",
    end_zone: "Penalty area",
    phase: "0-15",
    shot_location_x: 109,
    shot_location_y: 39,
    pass_location_x: 120,
    pass_location_y: 64,
    pass_end_location_x: 109,
    pass_end_location_y: 39,
  },
  {
    match_id: "1",
    Match: "Malmö FF - AIK",
    team: "AIK",
    Minute: 28,
    Second: 2,
    Taker: "Player C",
    Shooter: "Player D",
    set_piece_type: "Free Kick",
    shot_xg: 0.07,
    led_to_shot: true,
    goal: false,
    side: "Right",
    delivery_zone: "Near Post Zone",
    end_zone: "Deep box",
    phase: "16-30",
    shot_location_x: 103,
    shot_location_y: 25,
    pass_location_x: 92,
    pass_location_y: 21,
    pass_end_location_x: 103,
    pass_end_location_y: 25,
  },
  {
    match_id: "2",
    Match: "Hammarby - Djurgården",
    team: "Hammarby",
    Minute: 53,
    Second: 44,
    Taker: "Player E",
    Shooter: "Player F",
    set_piece_type: "Throw-In",
    shot_xg: 0.18,
    led_to_shot: true,
    goal: true,
    side: "Left",
    delivery_zone: "Far Post Zone",
    end_zone: "6-yard box",
    phase: "46-60",
    shot_location_x: 116,
    shot_location_y: 52,
    pass_location_x: 96,
    pass_location_y: 62,
    pass_end_location_x: 116,
    pass_end_location_y: 52,
  },
  {
    match_id: "2",
    Match: "Hammarby - Djurgården",
    team: "Djurgården",
    Minute: 72,
    Second: 10,
    Taker: "Player G",
    Shooter: "",
    set_piece_type: "Corner",
    shot_xg: 0,
    led_to_shot: false,
    goal: false,
    side: "Right",
    delivery_zone: "Near Post Zone",
    end_zone: "Outside danger zone",
    phase: "61-75",
    pass_location_x: 120,
    pass_location_y: 18,
    pass_end_location_x: 101,
    pass_end_location_y: 24,
  },
  {
    match_id: "3",
    Match: "Elfsborg - IFK Göteborg",
    team: "Elfsborg",
    Minute: 81,
    Second: 5,
    Taker: "Player H",
    Shooter: "Player I",
    set_piece_type: "Free Kick",
    shot_xg: 0.22,
    led_to_shot: true,
    goal: true,
    side: "Left",
    delivery_zone: "Central Zone",
    end_zone: "6-yard box",
    phase: "76+",
    shot_location_x: 115,
    shot_location_y: 40,
    pass_location_x: 88,
    pass_location_y: 60,
    pass_end_location_x: 115,
    pass_end_location_y: 40,
  },
  {
    match_id: "3",
    Match: "Elfsborg - IFK Göteborg",
    team: "IFK Göteborg",
    Minute: 9,
    Second: 20,
    Taker: "Player J",
    Shooter: "",
    set_piece_type: "Throw-In",
    shot_xg: 0,
    led_to_shot: false,
    goal: false,
    side: "Right",
    delivery_zone: "Central Zone",
    end_zone: "Penalty area",
    phase: "0-15",
    pass_location_x: 94,
    pass_location_y: 17,
    pass_end_location_x: 108,
    pass_end_location_y: 37,
  },
];

function groupCount<T>(rows: T[], keyFn: (row: T) => string) {
  const map = new Map<string, number>();
  rows.forEach((row) => {
    const key = keyFn(row) || "Unknown";
    map.set(key, (map.get(key) || 0) + 1);
  });
  return Array.from(map.entries()).map(([name, value]) => ({ name, value }));
}

function mean(arr: number[]) {
  if (!arr.length) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function parseCsv(text: string): EventRow[] {
  const lines = text.split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) return [];

  const headers = lines[0].split(",").map((h) => h.trim());
  return lines.slice(1).map((line, idx) => {
    const values = line.split(",");
    const row: Record<string, string> = {};
    headers.forEach((h, i) => {
      row[h] = values[i] ?? "";
    });

    const spRaw = String(row.SP_Type || row.set_piece_type || "Corner");
    const set_piece_type: SetPieceType = spRaw.toLowerCase().includes("free")
      ? "Free Kick"
      : spRaw.toLowerCase().includes("throw")
        ? "Throw-In"
        : "Corner";

    return {
      match_id: row.match_id || String(idx + 1),
      Match: row.Match || row.match || `Match ${idx + 1}`,
      team: row.team || row["team.name"] || "Unknown",
      Minute: Number(row.Minute || row.minute || 0),
      Second: Number(row.Second || row.second || 0),
      Taker: row.Taker || row.taker || "",
      Shooter: row.Shooter || row.shooter || "",
      set_piece_type,
      shot_xg: Number(row.shot_xg || row["shot.statsbomb_xg"] || 0),
      led_to_shot: String(row.led_to_shot || row.SP_outcome || "").toLowerCase().includes("shot") || Number(row.shot_xg || 0) > 0,
      goal: String(row.goal || row.shot_outcome || "").toLowerCase().includes("goal"),
      side: (row.side as "Left" | "Right" | "Unknown") || "Unknown",
      delivery_zone: row.delivery_zone || "Unknown",
      end_zone: row.end_zone || "Unknown",
      phase: row.phase || "Unknown",
      shot_location_x: Number(row.shot_location_x || 0) || undefined,
      shot_location_y: Number(row.shot_location_y || 0) || undefined,
      pass_location_x: Number(row.pass_location_x || 0) || undefined,
      pass_location_y: Number(row.pass_location_y || 0) || undefined,
      pass_end_location_x: Number(row.pass_end_location_x || 0) || undefined,
      pass_end_location_y: Number(row.pass_end_location_y || 0) || undefined,
    };
  });
}

function KpiCard({ title, value, foot, icon }: { title: string; value: string; foot: string; icon: React.ReactNode }) {
  return (
    <Card className="border-white/10 bg-white/[0.03] backdrop-blur-sm rounded-3xl shadow-xl">
      <CardContent className="p-5">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.18em] text-slate-400 font-bold">{title}</p>
            <p className="mt-3 text-3xl font-black text-white">{value}</p>
            <p className="mt-2 text-sm text-slate-400">{foot}</p>
          </div>
          <div className="rounded-2xl border border-white/10 bg-white/5 p-3 text-slate-200">{icon}</div>
        </div>
      </CardContent>
    </Card>
  );
}

function SectionCard({ title, subtitle, children }: { title: string; subtitle?: string; children: React.ReactNode }) {
  return (
    <Card className="border-white/10 bg-white/[0.03] rounded-3xl shadow-xl">
      <CardHeader className="pb-2">
        <CardTitle className="text-white text-lg font-extrabold">{title}</CardTitle>
        {subtitle ? <p className="text-sm text-slate-400">{subtitle}</p> : null}
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  );
}

function Landing({ data, onOpen }: { data: EventRow[]; onOpen: (segment: SetPieceType) => void }) {
  const summary = useMemo(() => {
    return (Object.keys(TYPE_META) as SetPieceType[]).map((type) => {
      const rows = data.filter((r) => r.set_piece_type === type);
      return {
        type,
        events: rows.length,
        matches: new Set(rows.map((r) => r.match_id)).size,
        shotRate: rows.length ? rows.filter((r) => r.led_to_shot).length / rows.length : 0,
        xgPerEvent: rows.length ? rows.reduce((s, r) => s + r.shot_xg, 0) / rows.length : 0,
      };
    });
  }, [data]);

  return (
    <div className="space-y-8">
      <motion.div
        initial={{ opacity: 0, y: 18 }}
        animate={{ opacity: 1, y: 0 }}
        className="rounded-[2rem] border border-sky-400/20 bg-gradient-to-br from-sky-400/15 via-sky-400/5 to-emerald-400/10 p-8 shadow-2xl"
      >
        <div className="max-w-4xl">
          <p className="mb-3 inline-flex rounded-full border border-white/10 bg-white/5 px-4 py-1 text-xs font-bold uppercase tracking-[0.18em] text-slate-300">
            New build
          </p>
          <h1 className="text-5xl font-black tracking-tight text-white">
            Allsvenskan <span className="text-sky-300">Set Piece</span> Studio
          </h1>
          <p className="mt-4 text-lg leading-8 text-slate-300">
            A totally new web app built around one simple landing page: choose <strong>Free Kick</strong>, <strong>Corner</strong>, or <strong>Throw-In</strong> and enter a focused analysis workspace.
          </p>
        </div>
      </motion.div>

      <div className="grid gap-5 md:grid-cols-3">
        {(Object.keys(TYPE_META) as SetPieceType[]).map((type, i) => {
          const meta = TYPE_META[type];
          const row = summary.find((s) => s.type === type)!;
          return (
            <motion.div
              key={type}
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.06 }}
            >
              <Card className={`h-full rounded-[2rem] border-white/10 bg-gradient-to-br ${meta.color} shadow-xl`}>
                <CardContent className="flex h-full flex-col p-6">
                  <div className="mb-4 flex items-center justify-between">
                    <Badge className="rounded-full border-white/10 bg-white/10 text-white">Segment</Badge>
                    <div className="rounded-2xl border border-white/10 bg-white/10 p-3 text-white">{meta.icon}</div>
                  </div>
                  <h2 className="text-2xl font-black text-white">{type}</h2>
                  <p className="mt-3 min-h-[72px] text-sm leading-6 text-slate-300">{meta.desc}</p>
                  <div className="mt-5 grid grid-cols-2 gap-3 text-sm">
                    <div className="rounded-2xl border border-white/10 bg-black/10 p-3">
                      <p className="text-slate-400">Events</p>
                      <p className="mt-1 text-2xl font-black text-white">{row.events}</p>
                    </div>
                    <div className="rounded-2xl border border-white/10 bg-black/10 p-3">
                      <p className="text-slate-400">Shot rate</p>
                      <p className="mt-1 text-2xl font-black text-white">{(row.shotRate * 100).toFixed(0)}%</p>
                    </div>
                  </div>
                  <Button className="mt-6 rounded-2xl bg-white/10 hover:bg-white/20" onClick={() => onOpen(type)}>
                    Open {type}
                  </Button>
                </CardContent>
              </Card>
            </motion.div>
          );
        })}
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <SectionCard title="Volume by segment" subtitle="Top-level navigation starts here">
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={summary}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                <XAxis dataKey="type" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip />
                <Bar dataKey="events" radius={[10, 10, 0, 0]}>
                  {summary.map((s) => (
                    <Cell key={s.type} fill={TYPE_META[s.type].accent === "bg-sky-400" ? "#38bdf8" : TYPE_META[s.type].accent === "bg-emerald-400" ? "#34d399" : "#fb923c"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </SectionCard>

        <SectionCard title="xG per event" subtitle="Quick efficiency comparison">
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={summary}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                <XAxis dataKey="type" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip />
                <Bar dataKey="xgPerEvent" fill="#c084fc" radius={[10, 10, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </SectionCard>
      </div>
    </div>
  );
}

function SegmentStudio({
  data,
  segment,
  onBack,
}: {
  data: EventRow[];
  segment: SetPieceType;
  onBack: () => void;
}) {
  const [team, setTeam] = useState<string>("all");
  const [side, setSide] = useState<string>("all");
  const [match, setMatch] = useState<string>("all");
  const [search, setSearch] = useState("");

  const segmentRows = useMemo(() => data.filter((r) => r.set_piece_type === segment), [data, segment]);

  const filtered = useMemo(() => {
    return segmentRows.filter((r) => {
      const okTeam = team === "all" || r.team === team;
      const okSide = side === "all" || r.side === side;
      const okMatch = match === "all" || r.Match === match;
      const needle = search.toLowerCase();
      const okSearch = !needle || [r.team, r.Taker, r.Shooter, r.Match].join(" ").toLowerCase().includes(needle);
      return okTeam && okSide && okMatch && okSearch;
    });
  }, [segmentRows, team, side, match, search]);

  const teams = Array.from(new Set(segmentRows.map((r) => r.team))).sort();
  const matches = Array.from(new Set(segmentRows.map((r) => r.Match))).sort();

  const kpis = useMemo(() => {
    const shots = filtered.filter((r) => r.led_to_shot).length;
    const goals = filtered.filter((r) => r.goal).length;
    const xg = filtered.reduce((s, r) => s + r.shot_xg, 0);
    return {
      events: filtered.length,
      matches: new Set(filtered.map((r) => r.match_id)).size,
      shots,
      goals,
      shotRate: filtered.length ? shots / filtered.length : 0,
      xg,
    };
  }, [filtered]);

  const endZones = groupCount(filtered, (r) => r.end_zone);
  const phases = groupCount(filtered, (r) => r.phase);
  const teamsTable = useMemo(() => {
    const map = new Map<string, { team: string; events: number; shots: number; xg: number }>();
    filtered.forEach((r) => {
      const cur = map.get(r.team) || { team: r.team, events: 0, shots: 0, xg: 0 };
      cur.events += 1;
      cur.shots += r.led_to_shot ? 1 : 0;
      cur.xg += r.shot_xg;
      map.set(r.team, cur);
    });
    return Array.from(map.values())
      .map((r) => ({ ...r, shotRate: r.events ? r.shots / r.events : 0, xgPerEvent: r.events ? r.xg / r.events : 0 }))
      .sort((a, b) => b.xgPerEvent - a.xgPerEvent);
  }, [filtered]);

  const takerTable = useMemo(() => {
    const map = new Map<string, { taker: string; team: string; events: number; shots: number; xg: number }>();
    filtered.forEach((r) => {
      const key = `${r.team}__${r.Taker || "Unknown"}`;
      const cur = map.get(key) || { taker: r.Taker || "Unknown", team: r.team, events: 0, shots: 0, xg: 0 };
      cur.events += 1;
      cur.shots += r.led_to_shot ? 1 : 0;
      cur.xg += r.shot_xg;
      map.set(key, cur);
    });
    return Array.from(map.values())
      .map((r) => ({ ...r, shotRate: r.events ? r.shots / r.events : 0, xgPerEvent: r.events ? r.xg / r.events : 0 }))
      .sort((a, b) => b.events - a.events);
  }, [filtered]);

  const matchTable = useMemo(() => {
    const map = new Map<string, { match: string; events: number; shots: number; xg: number; goals: number }>();
    filtered.forEach((r) => {
      const cur = map.get(r.Match) || { match: r.Match, events: 0, shots: 0, xg: 0, goals: 0 };
      cur.events += 1;
      cur.shots += r.led_to_shot ? 1 : 0;
      cur.xg += r.shot_xg;
      cur.goals += r.goal ? 1 : 0;
      map.set(r.Match, cur);
    });
    return Array.from(map.values())
      .map((r) => ({ ...r, shotRate: r.events ? r.shots / r.events : 0 }))
      .sort((a, b) => b.xg - a.xg);
  }, [filtered]);

  const minuteTrend = useMemo(() => {
    const buckets = Array.from({ length: 6 }).map((_, i) => ({
      bucket: ["0-15", "16-30", "31-45", "46-60", "61-75", "76+"][i],
      events: phases.find((p) => p.name === ["0-15", "16-30", "31-45", "46-60", "61-75", "76+"][i])?.value || 0,
    }));
    return buckets;
  }, [phases]);

  const scatterData = teamsTable.map((r) => ({ x: r.shotRate, y: r.xgPerEvent, z: r.events, team: r.team }));

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 rounded-[2rem] border border-white/10 bg-gradient-to-br from-white/5 to-white/[0.02] p-6 shadow-2xl lg:flex-row lg:items-center lg:justify-between">
        <div>
          <Button variant="ghost" className="mb-4 rounded-2xl border border-white/10 text-slate-200 hover:bg-white/5" onClick={onBack}>
            <ChevronLeft className="mr-2 h-4 w-4" /> Back to landing
          </Button>
          <h1 className="text-4xl font-black text-white">{segment} Studio</h1>
          <p className="mt-2 max-w-2xl text-slate-300">
            A fresh workspace for {segment.toLowerCase()} analysis with filters, visuals, team rankings, taker profiles, and match context.
          </p>
        </div>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <Select value={team} onValueChange={setTeam}>
            <SelectTrigger className="rounded-2xl border-white/10 bg-white/5 text-white"><SelectValue placeholder="Team" /></SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All teams</SelectItem>
              {teams.map((t) => <SelectItem key={t} value={t}>{t}</SelectItem>)}
            </SelectContent>
          </Select>
          <Select value={side} onValueChange={setSide}>
            <SelectTrigger className="rounded-2xl border-white/10 bg-white/5 text-white"><SelectValue placeholder="Side" /></SelectTrigger>
            <SelectContent>
              <SelectItem value="all">Both sides</SelectItem>
              <SelectItem value="Left">Left</SelectItem>
              <SelectItem value="Right">Right</SelectItem>
              <SelectItem value="Unknown">Unknown</SelectItem>
            </SelectContent>
          </Select>
          <Select value={match} onValueChange={setMatch}>
            <SelectTrigger className="rounded-2xl border-white/10 bg-white/5 text-white"><SelectValue placeholder="Match" /></SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All matches</SelectItem>
              {matches.map((m) => <SelectItem key={m} value={m}>{m}</SelectItem>)}
            </SelectContent>
          </Select>
          <Input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search team, taker, shooter..."
            className="rounded-2xl border-white/10 bg-white/5 text-white placeholder:text-slate-400"
          />
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
        <KpiCard title="Events" value={String(kpis.events)} foot="Filtered view" icon={<Activity className="h-5 w-5" />} />
        <KpiCard title="Matches" value={String(kpis.matches)} foot="Unique matches" icon={<BarChart3 className="h-5 w-5" />} />
        <KpiCard title="Shots" value={String(kpis.shots)} foot="From set pieces" icon={<Crosshair className="h-5 w-5" />} />
        <KpiCard title="Goals" value={String(kpis.goals)} foot="Direct outcome" icon={<Goal className="h-5 w-5" />} />
        <KpiCard title="Shot rate" value={`${(kpis.shotRate * 100).toFixed(1)}%`} foot={`${kpis.xg.toFixed(2)} total xG`} icon={<Users className="h-5 w-5" />} />
      </div>

      <Tabs defaultValue="overview" className="space-y-5">
        <TabsList className="grid h-auto grid-cols-3 rounded-2xl border border-white/10 bg-white/5 p-1 md:grid-cols-6">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="visuals">Visuals</TabsTrigger>
          <TabsTrigger value="teams">Teams</TabsTrigger>
          <TabsTrigger value="takers">Takers</TabsTrigger>
          <TabsTrigger value="matches">Matches</TabsTrigger>
          <TabsTrigger value="data">Data</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-5">
          <div className="grid gap-5 xl:grid-cols-2">
            <SectionCard title="End-zone distribution" subtitle="Where the action finishes">
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={endZones}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                    <XAxis dataKey="name" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" />
                    <Tooltip />
                    <Bar dataKey="value" fill="#60a5fa" radius={[10, 10, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </SectionCard>

            <SectionCard title="Phase timing" subtitle="When the segment appears in matches">
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={minuteTrend}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                    <XAxis dataKey="bucket" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" />
                    <Tooltip />
                    <Line type="monotone" dataKey="events" stroke="#34d399" strokeWidth={3} dot={{ r: 5 }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </SectionCard>
          </div>

          <SectionCard title="Team efficiency map" subtitle="Shot rate against xG per event">
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                  <XAxis type="number" dataKey="x" name="Shot rate" stroke="#94a3b8" />
                  <YAxis type="number" dataKey="y" name="xG per event" stroke="#94a3b8" />
                  <Tooltip cursor={{ strokeDasharray: "3 3" }} formatter={(value: number) => value.toFixed(3)} />
                  <Scatter data={scatterData} fill="#c084fc" />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </SectionCard>
        </TabsContent>

        <TabsContent value="visuals" className="space-y-5">
          <div className="grid gap-5 xl:grid-cols-2">
            <SectionCard title="Shot map" subtitle="Shot locations from the current filtered view">
              <div className="space-y-3">
                <div className="rounded-3xl border border-white/10 bg-slate-950 p-4">
                  <div className="relative aspect-[4/5] overflow-hidden rounded-2xl border border-white/10 bg-gradient-to-b from-slate-900 to-slate-950">
                    <div className="absolute inset-x-[10%] top-[70%] h-[18%] border border-white/50" />
                    <div className="absolute inset-x-[25%] top-[82%] h-[6%] border border-white/50" />
                    {filtered.filter((r) => r.shot_location_x && r.shot_location_y).map((r, i) => {
                      const left = Math.min(95, Math.max(5, ((r.shot_location_y || 40) / 80) * 100));
                      const top = Math.min(95, Math.max(5, 100 - ((r.shot_location_x || 100) / 120) * 100));
                      const size = 10 + r.shot_xg * 60;
                      return (
                        <div
                          key={i}
                          className="absolute rounded-full border border-white/80 bg-sky-400/70"
                          style={{ left: `${left}%`, top: `${top}%`, width: size, height: size, transform: "translate(-50%, -50%)" }}
                          title={`${r.team} | ${r.Shooter || "Unknown"} | xG ${r.shot_xg.toFixed(3)}`}
                        />
                      );
                    })}
                  </div>
                </div>
              </div>
            </SectionCard>

            <SectionCard title="Delivery map" subtitle="End locations of deliveries or actions">
              <div className="rounded-3xl border border-white/10 bg-slate-950 p-4">
                <div className="relative aspect-[4/5] overflow-hidden rounded-2xl border border-white/10 bg-gradient-to-b from-slate-900 to-slate-950">
                  <div className="absolute inset-0 border border-white/50" />
                  <div className="absolute inset-x-0 top-1/2 border-t border-white/30" />
                  <div className="absolute inset-x-[10%] top-[70%] h-[18%] border border-white/50" />
                  <div className="absolute inset-x-[25%] top-[82%] h-[6%] border border-white/50" />
                  {filtered.filter((r) => r.pass_end_location_x && r.pass_end_location_y).map((r, i) => {
                    const left = Math.min(95, Math.max(5, ((r.pass_end_location_y || 40) / 80) * 100));
                    const top = Math.min(95, Math.max(5, 100 - ((r.pass_end_location_x || 100) / 120) * 100));
                    return (
                      <div
                        key={i}
                        className="absolute h-3.5 w-3.5 rounded-full border border-white/80 bg-emerald-400/80"
                        style={{ left: `${left}%`, top: `${top}%`, transform: "translate(-50%, -50%)" }}
                        title={`${r.team} | ${r.Taker || "Unknown"} | ${r.delivery_zone}`}
                      />
                    );
                  })}
                </div>
              </div>
            </SectionCard>
          </div>
        </TabsContent>

        <TabsContent value="teams">
          <SectionCard title="Team rankings" subtitle="Best teams in the current segment view">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="border-b border-white/10 text-slate-400">
                  <tr>
                    <th className="px-3 py-3">Team</th>
                    <th className="px-3 py-3">Events</th>
                    <th className="px-3 py-3">Shots</th>
                    <th className="px-3 py-3">Shot rate</th>
                    <th className="px-3 py-3">xG/event</th>
                  </tr>
                </thead>
                <tbody>
                  {teamsTable.map((row) => (
                    <tr key={row.team} className="border-b border-white/5 text-white">
                      <td className="px-3 py-3 font-semibold">{row.team}</td>
                      <td className="px-3 py-3">{row.events}</td>
                      <td className="px-3 py-3">{row.shots}</td>
                      <td className="px-3 py-3">{(row.shotRate * 100).toFixed(1)}%</td>
                      <td className="px-3 py-3">{row.xgPerEvent.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>
        </TabsContent>

        <TabsContent value="takers">
          <SectionCard title="Taker profiles" subtitle="Usage and output by taker">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="border-b border-white/10 text-slate-400">
                  <tr>
                    <th className="px-3 py-3">Taker</th>
                    <th className="px-3 py-3">Team</th>
                    <th className="px-3 py-3">Events</th>
                    <th className="px-3 py-3">Shot rate</th>
                    <th className="px-3 py-3">xG/event</th>
                  </tr>
                </thead>
                <tbody>
                  {takerTable.map((row, i) => (
                    <tr key={`${row.team}-${row.taker}-${i}`} className="border-b border-white/5 text-white">
                      <td className="px-3 py-3 font-semibold">{row.taker}</td>
                      <td className="px-3 py-3">{row.team}</td>
                      <td className="px-3 py-3">{row.events}</td>
                      <td className="px-3 py-3">{(row.shotRate * 100).toFixed(1)}%</td>
                      <td className="px-3 py-3">{row.xgPerEvent.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>
        </TabsContent>

        <TabsContent value="matches">
          <SectionCard title="Match board" subtitle="Where the segment mattered most">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="border-b border-white/10 text-slate-400">
                  <tr>
                    <th className="px-3 py-3">Match</th>
                    <th className="px-3 py-3">Events</th>
                    <th className="px-3 py-3">Shots</th>
                    <th className="px-3 py-3">Goals</th>
                    <th className="px-3 py-3">xG</th>
                  </tr>
                </thead>
                <tbody>
                  {matchTable.map((row) => (
                    <tr key={row.match} className="border-b border-white/5 text-white">
                      <td className="px-3 py-3 font-semibold">{row.match}</td>
                      <td className="px-3 py-3">{row.events}</td>
                      <td className="px-3 py-3">{row.shots}</td>
                      <td className="px-3 py-3">{row.goals}</td>
                      <td className="px-3 py-3">{row.xg.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>
        </TabsContent>

        <TabsContent value="data" className="space-y-5">
          <SectionCard title="Raw segment data" subtitle="Current filtered rows">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="border-b border-white/10 text-slate-400">
                  <tr>
                    {[
                      "Match",
                      "team",
                      "Minute",
                      "Taker",
                      "Shooter",
                      "shot_xg",
                      "side",
                      "delivery_zone",
                      "end_zone",
                    ].map((h) => (
                      <th key={h} className="px-3 py-3">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((r, i) => (
                    <tr key={i} className="border-b border-white/5 text-white">
                      <td className="px-3 py-3">{r.Match}</td>
                      <td className="px-3 py-3">{r.team}</td>
                      <td className="px-3 py-3">{r.Minute}</td>
                      <td className="px-3 py-3">{r.Taker}</td>
                      <td className="px-3 py-3">{r.Shooter}</td>
                      <td className="px-3 py-3">{r.shot_xg.toFixed(3)}</td>
                      <td className="px-3 py-3">{r.side}</td>
                      <td className="px-3 py-3">{r.delivery_zone}</td>
                      <td className="px-3 py-3">{r.end_zone}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </SectionCard>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default function AllsvenskanSetPieceStudioApp() {
  const [segment, setSegment] = useState<SetPieceType | null>(null);
  const [rows, setRows] = useState<EventRow[]>(demoData);

  const handleUpload = async (file: File | null) => {
    if (!file) return;
    const text = await file.text();
    const parsed = parseCsv(text);
    if (parsed.length) setRows(parsed);
  };

  return (
    <div className="min-h-screen bg-[#07111f] text-white">
      <div className="mx-auto max-w-7xl p-6 md:p-8">
        <div className="mb-6 flex flex-col gap-4 rounded-[2rem] border border-white/10 bg-white/[0.03] p-5 shadow-xl md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.2em] text-slate-400">Fresh app build</p>
            <h1 className="mt-1 text-2xl font-black">Set Piece Analysis Platform</h1>
          </div>
          <label className="flex cursor-pointer items-center gap-3 rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-slate-200 hover:bg-white/10">
            <Upload className="h-4 w-4" />
            Upload CSV dataset
            <input type="file" accept=".csv" className="hidden" onChange={(e) => handleUpload(e.target.files?.[0] || null)} />
          </label>
        </div>

        {segment === null ? (
          <Landing data={rows} onOpen={setSegment} />
        ) : (
          <SegmentStudio data={rows} segment={segment} onBack={() => setSegment(null)} />
        )}
      </div>
    </div>
  );
}
