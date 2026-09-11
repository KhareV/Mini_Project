<script lang="ts">
	import { onMount } from 'svelte';
	import { Play, Pause, RotateCcw, SkipForward, SkipBack, ShieldCheck, TriangleAlert, Maximize2, Minimize2, Activity, Waves, BrainCircuit, BellRing } from '@lucide/svelte';

	type Phase = {
		name: string; short: string; duration: number; hr: number; spo2: number;
		ecgQuality: number; ppgQuality: number; ecgRisk: number; decision: string;
		tone: 'ok' | 'warn' | 'danger'; narrative: string;
	};

	const phases: Phase[] = [
		{ name: 'Stable baseline', short: 'BASELINE', duration: 18, hr: 74, spo2: 98, ecgQuality: 94, ppgQuality: 92, ecgRisk: .12, decision: 'NORMAL MONITORED PATTERN', tone: 'ok', narrative: 'Both sensors are stable. ECG and pulse agree, saturation is normal, and all evidence clears the quality gate.' },
		{ name: 'Activity transition', short: 'ACTIVITY', duration: 18, hr: 92, spo2: 97, ecgQuality: 82, ppgQuality: 76, ecgRisk: .24, decision: 'NORMAL / RATE ELEVATED', tone: 'ok', narrative: 'Pulse rises gradually while signal quality remains usable. The system preserves the trend without opening an alert.' },
		{ name: 'Motion artefact', short: 'LOW SQI', duration: 16, hr: 108, spo2: 95, ecgQuality: 38, ppgQuality: 31, ecgRisk: .71, decision: 'UNRELIABLE SIGNAL', tone: 'warn', narrative: 'Motion degrades both channels. Although the raw score rises, the quality gate suppresses a potentially false clinical-looking prediction.' },
		{ name: 'Abnormal vitals', short: 'ALERT', duration: 22, hr: 112, spo2: 92, ecgQuality: 88, ppgQuality: 84, ecgRisk: .79, decision: 'POTENTIALLY ABNORMAL', tone: 'danger', narrative: 'Quality recovers while pulse remains above 100 BPM and measured SpO₂ is below 95%. Independent evidence opens an alert.' },
		{ name: 'Recovery', short: 'RECOVERY', duration: 20, hr: 80, spo2: 97, ecgQuality: 93, ppgQuality: 90, ecgRisk: .21, decision: 'RECOVERING / NO NEW ALERT', tone: 'ok', narrative: 'Pulse and saturation return toward baseline. Consecutive reliable windows allow the event state to clear.' }
	];

	const totalDuration = phases.reduce((sum, phase) => sum + phase.duration, 0);
	let running = $state(false);
	let elapsed = $state(0);
	let speed = $state(1);
	let loops = $state(0);
	let lastPhase = $state(-1);
	let fullscreen = $state(false);
	let eventLog = $state<Array<{ time: number; label: string; tone: string }>>([]);

	const phaseIndex = $derived.by(() => {
		let cursor = 0;
		for (let index = 0; index < phases.length; index += 1) {
			cursor += phases[index].duration;
			if (elapsed < cursor) return index;
		}
		return phases.length - 1;
	});
	const phaseStart = $derived(phases.slice(0, phaseIndex).reduce((sum, phase) => sum + phase.duration, 0));
	const phaseProgress = $derived(Math.min(1, Math.max(0, (elapsed - phaseStart) / phases[phaseIndex].duration)));
	const phase = $derived(phases[phaseIndex]);
	const previous = $derived(phases[Math.max(0, phaseIndex - 1)]);
	const blend = $derived(Math.min(1, phaseProgress * 2.4));
	const hr = $derived(previous.hr + (phase.hr - previous.hr) * blend);
	const spo2 = $derived(previous.spo2 + (phase.spo2 - previous.spo2) * blend);
	const ecgQuality = $derived(previous.ecgQuality + (phase.ecgQuality - previous.ecgQuality) * blend);
	const ppgQuality = $derived(previous.ppgQuality + (phase.ppgQuality - previous.ppgQuality) * blend);
	const risk = $derived(previous.ecgRisk + (phase.ecgRisk - previous.ecgRisk) * blend);
	const overallProgress = $derived((elapsed / totalDuration) * 100);
	const qualityGate = $derived(Math.min(ecgQuality, ppgQuality) >= 50);
	const eventState = $derived(phaseIndex === 3 ? 'OPEN' : phaseIndex === 4 && phaseProgress < .45 ? 'OBSERVING' : 'CLEAR');

	function wavePoints(kind: 'ecg' | 'ppg', width = 900, height = 150) {
		const points: string[] = [];
		const rate = Math.max(52, hr);
		const cycles = rate / 10;
		for (let i = 0; i <= 220; i += 1) {
			const x = (i / 220) * width;
			const cycle = ((i / 220) * cycles) % 1;
			let signal = 0;
			if (kind === 'ecg') {
				signal = .05 * Math.sin(cycle * Math.PI * 2);
				if (cycle > .43 && cycle < .47) signal -= .18;
				if (cycle >= .47 && cycle < .50) signal += .92;
				if (cycle >= .50 && cycle < .54) signal -= .34;
				if (cycle > .66 && cycle < .82) signal += .16 * Math.sin(((cycle - .66) / .16) * Math.PI);
			} else {
				signal = cycle < .18 ? Math.sin((cycle / .18) * Math.PI) * .72 : Math.exp(-(cycle - .18) * 4.1) * .58;
				if (cycle > .42 && cycle < .58) signal += .12 * Math.sin(((cycle - .42) / .16) * Math.PI);
			}
			const quality = kind === 'ecg' ? ecgQuality : ppgQuality;
			const noise = (100 - quality) / 100 * (Math.sin(i * 12.9898 + elapsed * 2.1) * .34);
			const y = height * .58 - (signal + noise) * height * .43;
			points.push(`${x.toFixed(1)},${y.toFixed(1)}`);
		}
		return points.join(' ');
	}

	function jumpTo(index: number) {
		elapsed = phases.slice(0, index).reduce((sum, item) => sum + item.duration, 0) + .01;
	}
	function reset() { running = false; elapsed = 0; loops = 0; lastPhase = -1; eventLog = []; }
	function next() { jumpTo((phaseIndex + 1) % phases.length); }
	function previousPhase() { jumpTo((phaseIndex - 1 + phases.length) % phases.length); }
	async function toggleFullscreen() {
		if (document.fullscreenElement) await document.exitFullscreen();
		else await document.documentElement.requestFullscreen();
	}

	onMount(() => {
		const onFullscreenChange = () => fullscreen = Boolean(document.fullscreenElement);
		const onKeyDown = (event: KeyboardEvent) => {
			const target = event.target as HTMLElement | null;
			if (target?.matches('input, select, textarea')) return;
			if (event.code === 'Space') { event.preventDefault(); running = !running; }
			if (event.key === 'ArrowRight') { event.preventDefault(); next(); }
			if (event.key === 'ArrowLeft') { event.preventDefault(); previousPhase(); }
			if (event.key.toLowerCase() === 'r') reset();
			if (event.key.toLowerCase() === 'f') void toggleFullscreen();
		};
		document.addEventListener('fullscreenchange', onFullscreenChange);
		window.addEventListener('keydown', onKeyDown);
		const timer = window.setInterval(() => {
			if (!running) return;
			elapsed += .1 * speed;
			if (elapsed >= totalDuration) { elapsed = 0; loops += 1; lastPhase = -1; eventLog = []; }
			if (phaseIndex !== lastPhase) {
				lastPhase = phaseIndex;
				eventLog = [{ time: Math.round(elapsed), label: phases[phaseIndex].name, tone: phases[phaseIndex].tone }, ...eventLog].slice(0, 6);
			}
		}, 100);
		return () => {
			window.clearInterval(timer);
			document.removeEventListener('fullscreenchange', onFullscreenChange);
			window.removeEventListener('keydown', onKeyDown);
		};
	});
</script>

<svelte:head><title>Reviewer Simulation | NHM</title><meta name="description" content="Interactive simulated progression through the NHM monitoring pipeline." /></svelte:head>

<div class="demo-shell" class:danger={phase.tone === 'danger'} class:warning={phase.tone === 'warn'}>
	<div class="ambient" aria-hidden="true"></div>
	<header class="hero">
		<div><span class="kicker"><i></i> SIMULATED REVIEWER MODE</span><h1>Watch the system<br />reason over time.</h1><p>A deterministic frontend demonstration of signal progression, quality gating, vital alerts, and recovery. No patient or hardware data is used.</p></div>
		<div class="run-state"><span>SCENARIO</span><strong>DEMO-{String(loops + 1).padStart(2, '0')}</strong><small>{running ? 'RUNNING' : 'PAUSED'} · {speed}× SPEED</small></div>
	</header>

	<section class="controls" aria-label="Simulation controls">
		<button class="primary" onclick={() => running = !running}>{#if running}<Pause size={15} /> PAUSE{:else}<Play size={15} /> PLAY{/if}</button>
		<button onclick={previousPhase} aria-label="Previous phase"><SkipBack size={15} /> PREVIOUS</button>
		<button onclick={next}><SkipForward size={15} /> NEXT PHASE</button>
		<button onclick={reset}><RotateCcw size={15} /> RESET</button>
		<button onclick={toggleFullscreen}>{#if fullscreen}<Minimize2 size={15} /> EXIT FULLSCREEN{:else}<Maximize2 size={15} /> PRESENT{/if}</button>
		<label>SPEED <select bind:value={speed}><option value={.5}>0.5×</option><option value={1}>1×</option><option value={2}>2×</option><option value={4}>4×</option></select></label>
		<span class="clock">T+{Math.floor(elapsed / 60).toString().padStart(2, '0')}:{Math.floor(elapsed % 60).toString().padStart(2, '0')}</span>
	</section>
	<div class="shortcuts"><span><kbd>SPACE</kbd> PLAY / PAUSE</span><span><kbd>←</kbd><kbd>→</kbd> PHASE</span><span><kbd>F</kbd> PRESENT</span><span><kbd>R</kbd> RESET</span></div>

	<section class="timeline">
		<div class="progress"><i style={`width:${overallProgress}%`}></i></div>
		<div class="phase-row">{#each phases as item, index}<button class:active={index === phaseIndex} class:passed={index < phaseIndex} onclick={() => jumpTo(index)}><span>0{index + 1}</span><b>{item.short}</b><small>{item.duration}s</small></button>{/each}</div>
	</section>

	<section class="metric-grid">
		<article><span>PPG PULSE</span><strong>{Math.round(hr)}<small>BPM</small></strong><footer>V3 ESTIMATOR · {ppgQuality.toFixed(0)}% QUALITY</footer></article>
		<article><span>MEASURED SpO₂</span><strong>{spo2.toFixed(1)}<small>%</small></strong><footer>SENSOR VALUE · QUALITY TAGGED</footer></article>
		<article><span>ECG PATTERN SCORE</span><strong>{risk.toFixed(2)}<small>/ 1</small></strong><footer>MODEL_V1 · THRESHOLD 0.37</footer></article>
		<article class:alert={eventState === 'OPEN'}><span>EVENT STATE</span><strong class="state">{eventState}</strong><footer>2-WINDOW OPEN / CLEAR POLICY</footer></article>
	</section>

	<section class="pipeline" aria-label="System reasoning pipeline">
		<div class:active={phaseIndex >= 0}><Activity size={17} /><span>01 · SENSE</span><b>ECG + PPG</b></div><i>→</i>
		<div class:active={phaseIndex >= 1}><Waves size={17} /><span>02 · QUALIFY</span><b>SQI GATE</b></div><i>→</i>
		<div class:active={phaseIndex >= 2}><BrainCircuit size={17} /><span>03 · INFER</span><b>MODEL + RULES</b></div><i>→</i>
		<div class:active={phaseIndex >= 3}><BellRing size={17} /><span>04 · ACT</span><b>EVENT POLICY</b></div>
	</section>

	<section class="main-grid">
		<div class="signals">
			<article class="signal"><header><div><span>ECG / ELECTRICAL</span><b>{Math.round(hr)} BPM</b></div><small>AD8232 · 250 HZ · SQI {ecgQuality.toFixed(0)}%</small></header><svg viewBox="0 0 900 150" preserveAspectRatio="none" aria-label="Simulated ECG waveform"><defs><linearGradient id="ecg-fade"><stop stop-color="#fb7185" /><stop offset="1" stop-color="#f43f5e" /></linearGradient></defs><polyline points={wavePoints('ecg')} /></svg></article>
			<article class="signal ppg"><header><div><span>PPG / OPTICAL</span><b>{Math.round(hr)} BPM</b></div><small>MAX30102 · 125 HZ · SQI {ppgQuality.toFixed(0)}%</small></header><svg viewBox="0 0 900 150" preserveAspectRatio="none" aria-label="Simulated PPG waveform"><polyline points={wavePoints('ppg')} /></svg></article>
		</div>
		<aside class="reasoning">
			<div class="decision" class:blocked={!qualityGate}><span>{qualityGate ? 'SYSTEM DECISION' : 'QUALITY GATE'}</span>{#if qualityGate}<ShieldCheck size={24} />{:else}<TriangleAlert size={24} />{/if}<strong>{phase.decision}</strong><p>{phase.narrative}</p></div>
			<div class="quality-bars"><span>MODALITY QUALITY</span><div class="qrow">ECG <i><b style={`width:${ecgQuality}%`}></b></i><em>{ecgQuality.toFixed(0)}%</em></div><div class="qrow">PPG <i><b style={`width:${ppgQuality}%`}></b></i><em>{ppgQuality.toFixed(0)}%</em></div><div class="qrow">FUSION <i><b style={`width:${Math.min(ecgQuality, ppgQuality)}%`}></b></i><em>{Math.min(ecgQuality, ppgQuality).toFixed(0)}%</em></div></div>
			<div class="events"><span>PROGRESSION LOG</span>{#if eventLog.length}{#each eventLog as event}<div><i class={event.tone}></i><b>T+{event.time}s</b><small>{event.label}</small></div>{/each}{:else}<p>Press play to begin the scripted progression.</p>{/if}</div>
		</aside>
	</section>

	<section class="evidence">
		<header><div><span>REAL VALIDATION EVIDENCE</span><b>Measured results — separate from the scripted scenario above</b></div><a href="/ai/model">OPEN MODEL CARD ↗</a></header>
		<div class="evidence-grid">
			<article><span>ECG · PTB-XL HOLDOUT</span><strong>0.8389</strong><small>F1 · AUROC 0.9010</small></article>
			<article><span>PPG PULSE · BIDMC</span><strong>2.346</strong><small>BPM MAE · ALL WINDOWS</small></article>
			<article><span>PPG PULSE · QUALITY GATED</span><strong>1.810</strong><small>BPM MAE · ACCEPTED WINDOWS</small></article>
			<article><span>DEMO SAFETY</span><strong>0</strong><small>PATIENT RECORDS USED</small></article>
		</div>
	</section>

	<footer class="disclaimer"><b>SIMULATION ONLY</b><span>Values are scripted for demonstration. Not patient data, not a diagnosis, and not evidence of clinical performance.</span><a href="/ai/model">VIEW VALIDATED MODEL STATUS →</a></footer>
</div>

<style>
	.demo-shell{--accent:#2bb8b0;--accent-rgb:43,184,176;position:relative;isolation:isolate;max-width:1450px;margin:0 auto;padding:0 24px 28px;color:#dce9e8}.demo-shell.warning{--accent:#fbbf24;--accent-rgb:251,191,36}.demo-shell.danger{--accent:#fb7185;--accent-rgb:251,113,133}.ambient{position:fixed;z-index:-1;inset:-20%;pointer-events:none;background:radial-gradient(circle at 84% 20%,rgba(var(--accent-rgb),.10),transparent 28%),radial-gradient(circle at 15% 75%,rgba(14,165,233,.065),transparent 25%);transition:background .8s ease}.hero{display:flex;justify-content:space-between;gap:40px;padding:28px 0 42px}.kicker,.hero p,.run-state span,.run-state small,.controls,.shortcuts,.phase-row,.metric-grid span,.metric-grid footer,.signal header,.reasoning span,.pipeline,.evidence,.disclaimer{font-family:'JetBrains Mono',monospace}.kicker{color:var(--accent);font-size:8px;letter-spacing:.18em}.kicker i{display:inline-block;width:6px;height:6px;margin-right:8px;border-radius:50%;background:var(--accent);box-shadow:0 0 12px var(--accent);animation:pulse 1.8s infinite}.demo-shell h1{margin:17px 0 15px;font:500 clamp(38px,5vw,72px)/.92 'Space Grotesk',sans-serif;letter-spacing:-.045em}.hero p{max-width:670px;margin:0;color:#71829a;font-size:10px;line-height:1.7}.run-state{align-self:flex-end;min-width:190px;padding:18px;border-left:2px solid var(--accent);background:rgba(var(--accent-rgb),.06)}.run-state span,.run-state small{display:block;color:#64748b;font-size:8px;letter-spacing:.12em}.run-state strong{display:block;margin:9px 0;font:500 22px 'Space Grotesk',sans-serif}.controls{display:flex;align-items:center;gap:7px;flex-wrap:wrap;padding:11px;border:1px solid rgba(148,163,184,.14);background:#070d18;font-size:8px;letter-spacing:.08em}.controls button{display:flex;align-items:center;gap:7px;padding:9px 12px;border:1px solid #263546;color:#91a8b8;background:#0a1220;font:inherit;cursor:pointer}.controls button:hover,.controls .primary{border-color:var(--accent);color:#03110f;background:var(--accent)}.controls label{display:flex;align-items:center;gap:8px;margin-left:auto;color:#64748b}.controls select{padding:7px;border:1px solid #263546;color:#dce9e8;background:#0a1220;font:inherit}.clock{min-width:54px;color:var(--accent);text-align:right}.shortcuts{display:flex;justify-content:flex-end;gap:15px;padding:8px 2px;color:#405168;font-size:6px;letter-spacing:.08em}.shortcuts kbd{padding:2px 4px;border:1px solid #263546;color:#71829a;background:#080f1b;font:inherit}.timeline{margin:4px 0 12px}.progress{height:2px;background:#172434}.progress i{display:block;height:100%;background:var(--accent);box-shadow:0 0 10px var(--accent);transition:width .1s linear}.phase-row{display:grid;grid-template-columns:repeat(5,1fr)}.phase-row button{display:grid;grid-template-columns:auto 1fr auto;gap:8px;padding:12px 9px;border:0;border-right:1px solid rgba(148,163,184,.1);color:#53647b;background:transparent;text-align:left;font:inherit;cursor:pointer}.phase-row button.active{color:var(--accent);background:rgba(var(--accent-rgb),.06)}.phase-row button.passed{color:#91a8b8}.phase-row span,.phase-row small{font-size:7px}.phase-row b{font-size:8px;letter-spacing:.08em}.metric-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:7px;margin:18px 0 7px}.metric-grid article{min-height:126px;padding:16px;border:1px solid rgba(148,163,184,.14);background:#080f1b;transition:border-color .5s,background .5s}.metric-grid article.alert{border-color:rgba(251,113,133,.45);background:rgba(127,29,29,.14);box-shadow:inset 0 0 45px rgba(251,113,133,.05)}.metric-grid span,.metric-grid footer{display:block;color:#53647b;font-size:7px;letter-spacing:.12em}.metric-grid strong{display:block;margin:19px 0 18px;font:500 34px 'Space Grotesk',sans-serif}.metric-grid strong small{margin-left:6px;color:#64748b;font:8px 'JetBrains Mono',monospace}.metric-grid .state{color:var(--accent);font-size:27px}.pipeline{display:flex;align-items:stretch;gap:8px;margin:0 0 7px;padding:10px;border:1px solid rgba(148,163,184,.12);background:#050b14}.pipeline>div{display:grid;grid-template-columns:24px 1fr;grid-template-rows:auto auto;align-items:center;flex:1;padding:8px;color:#405168;border:1px solid rgba(148,163,184,.08);transition:.45s ease}.pipeline>div.active{color:var(--accent);border-color:rgba(var(--accent-rgb),.22);background:rgba(var(--accent-rgb),.04)}.pipeline svg{grid-row:span 2}.pipeline span{font-size:6px;letter-spacing:.12em}.pipeline b{color:#91a8b8;font-size:7px}.pipeline>i{align-self:center;color:#263546;font-style:normal}.main-grid{display:grid;grid-template-columns:minmax(0,1.65fr) minmax(280px,.55fr);gap:7px}.signals{display:grid;gap:7px}.signal{overflow:hidden;border:1px solid rgba(148,163,184,.14);background:linear-gradient(135deg,rgba(251,113,133,.045),#070d18 55%)}.signal.ppg{background:linear-gradient(135deg,rgba(56,189,248,.045),#070d18 55%)}.signal header{display:flex;justify-content:space-between;padding:14px 16px;color:#53647b;font-size:7px;letter-spacing:.1em}.signal header div{display:flex;gap:16px}.signal header b{color:#dce9e8}.signal svg{display:block;width:100%;height:165px;background-image:linear-gradient(rgba(148,163,184,.035) 1px,transparent 1px),linear-gradient(90deg,rgba(148,163,184,.035) 1px,transparent 1px);background-size:22px 22px}.signal polyline{fill:none;stroke:url(#ecg-fade);stroke-width:1.8;vector-effect:non-scaling-stroke;filter:drop-shadow(0 0 4px rgba(251,113,133,.35));transition:stroke .5s}.signal.ppg polyline{stroke:#38bdf8;filter:drop-shadow(0 0 4px rgba(56,189,248,.35))}.reasoning{display:grid;gap:7px}.reasoning>div{padding:16px;border:1px solid rgba(148,163,184,.14);background:#080f1b}.decision{color:var(--accent)}.decision.blocked{color:#fbbf24}.decision>span{display:inline-block;margin-right:8px;color:#64748b;font-size:7px;letter-spacing:.13em;vertical-align:7px}.decision strong{display:block;margin:15px 0 9px;font:500 20px/1.1 'Space Grotesk',sans-serif}.decision p,.events p{margin:0;color:#71829a;font-size:10px;line-height:1.6}.quality-bars>span,.events>span{display:block;margin-bottom:15px;color:#64748b;font-size:7px;letter-spacing:.13em}.quality-bars .qrow{display:grid;grid-template-columns:42px 1fr 34px;align-items:center;gap:8px;margin:10px 0;color:#71829a;font:7px 'JetBrains Mono',monospace}.quality-bars .qrow>i{height:3px;background:#172434}.quality-bars .qrow b{display:block;height:100%;background:var(--accent);transition:width .3s}.quality-bars em{font-style:normal;text-align:right}.events div{display:grid;grid-template-columns:8px 38px 1fr;align-items:center;gap:6px;margin:10px 0;font:7px 'JetBrains Mono',monospace}.events div i{width:5px;height:5px;border-radius:50%;background:#2bb8b0}.events div i.warn{background:#fbbf24}.events div i.danger{background:#fb7185}.events div b{color:#53647b}.events div small{color:#91a8b8}.evidence{margin-top:7px;border:1px solid rgba(43,184,176,.2);background:linear-gradient(135deg,rgba(43,184,176,.045),#070d18 42%)}.evidence>header{display:flex;justify-content:space-between;align-items:center;padding:13px 15px;border-bottom:1px solid rgba(148,163,184,.1)}.evidence>header div{display:grid;gap:4px}.evidence>header span{color:#2bb8b0;font-size:7px;letter-spacing:.14em}.evidence>header b{color:#64748b;font-size:7px;font-weight:400}.evidence>header a{color:#2bb8b0;font-size:7px;text-decoration:none}.evidence-grid{display:grid;grid-template-columns:repeat(4,1fr)}.evidence-grid article{padding:16px;border-right:1px solid rgba(148,163,184,.1)}.evidence-grid span,.evidence-grid small{display:block;color:#53647b;font-size:6px;letter-spacing:.11em}.evidence-grid strong{display:block;margin:10px 0 6px;color:#dce9e8;font:500 25px 'Space Grotesk',sans-serif}.disclaimer{display:flex;gap:14px;align-items:center;margin-top:18px;padding:14px;border:1px solid rgba(251,191,36,.22);color:#71829a;font-size:7px;letter-spacing:.08em}.disclaimer b{color:#fbbf24}.disclaimer a{margin-left:auto;color:#2bb8b0;text-decoration:none}@keyframes pulse{50%{opacity:.45;box-shadow:0 0 22px var(--accent)}}@media(max-width:950px){.main-grid{grid-template-columns:1fr}.reasoning{grid-template-columns:1fr 1fr}.events{grid-column:span 2}.metric-grid,.evidence-grid{grid-template-columns:1fr 1fr}.pipeline>i{display:none}.pipeline{display:grid;grid-template-columns:1fr 1fr}}@media(max-width:680px){.demo-shell{padding:0 12px 20px}.hero{display:block}.run-state{margin-top:24px}.phase-row{grid-template-columns:1fr}.phase-row button{display:flex}.metric-grid,.evidence-grid{grid-template-columns:1fr 1fr}.reasoning{grid-template-columns:1fr}.events{grid-column:auto}.signal header{display:block}.signal header small{display:block;margin-top:7px}.disclaimer,.evidence>header{align-items:flex-start;flex-direction:column}.disclaimer a{margin-left:0}.controls label{margin-left:0}.shortcuts{display:none}.pipeline{grid-template-columns:1fr}.evidence-grid article{border-bottom:1px solid rgba(148,163,184,.1)}}
</style>
