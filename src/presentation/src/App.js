import React, { useState, useEffect, useRef, useCallback } from 'react';
import './App.css';
import SpeechInput from './components/SpeechInput';
import StatusIndicator from './components/StatusIndicator';
import FeedbackDisplay from './components/FeedbackDisplay';
import ErrorFeedback from './utils/ErrorFeedback';
import UiClient from './utils/UiClient';
import AuditLogger from './utils/auditLogger';
import AccessibilityLayer from './utils/accessibilityLayer';

const AUDIO_BACKEND_BASE = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws';
const DEFAULT_TRAINING_THRESHOLD_SEC = 1800; // 30 minutes (fallback if API call fails)

// ---------------------------------------------------------------------------
// WebSocket hook — single persistent connection with exponential backoff
// reconnection. Replaces all setInterval polling.
// ---------------------------------------------------------------------------
function useServerSocket(onMessage) {
  const wsRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const attemptsRef = useRef(0);
  const mountedRef = useRef(true);
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;

  const connect = useCallback(() => {
    if (!mountedRef.current) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      attemptsRef.current = 0;
      // Send a ping every 20 s to keep the connection alive through proxies
      ws._pingInterval = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'ping' }));
        }
      }, 20_000);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'pong') return; // ignore keepalive replies
        onMessageRef.current(data);
      } catch { /* malformed frame — ignore */ }
    };

    ws.onclose = () => {
      clearInterval(ws._pingInterval);
      if (!mountedRef.current) return;
      // Exponential backoff: 500 ms, 1 s, 2 s, 4 s … capped at 16 s
      const delay = Math.min(500 * 2 ** attemptsRef.current, 16_000);
      attemptsRef.current += 1;
      reconnectTimerRef.current = setTimeout(connect, delay);
    };

    ws.onerror = () => {
      ws.close(); // triggers onclose → reconnect
    };
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    connect();
    return () => {
      mountedRef.current = false;
      clearTimeout(reconnectTimerRef.current);
      clearInterval(wsRef.current?._pingInterval);
      wsRef.current?.close();
    };
  }, [connect]);
}

// ---------------------------------------------------------------------------
// Preferences hook — still uses REST (write path); no polling needed here
// ---------------------------------------------------------------------------
function usePreferences(addToast) {
  const [prefs, setPrefs] = useState(null);

  const load = useCallback(async () => {
    try {
      const r = await fetch(`${AUDIO_BACKEND_BASE}/preferences`);
      if (r.ok) setPrefs(await r.json());
    } catch { /* silent */ }
  }, []);

  useEffect(() => { load(); }, [load]);

  const save = useCallback(async (updates) => {
    setPrefs(prev => ({ ...prev, ...updates }));
    try {
      const r = await fetch(`${AUDIO_BACKEND_BASE}/preferences`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updates),
      });
      if (!r.ok) throw new Error(`${r.status}`);
      setPrefs(await r.json());
    } catch (err) {
      load();
      addToast(err.message === '503'
        ? 'Server unavailable — preference not saved.'
        : 'Could not save preference. Please try again.');
    }
  }, [load, addToast]);

  return { prefs, save };
}

// ---------------------------------------------------------------------------
// Training config hook — fetches threshold from backend
// ---------------------------------------------------------------------------
function useTrainingConfig() {
  const [config, setConfig] = useState({ 
    audio_threshold_seconds: DEFAULT_TRAINING_THRESHOLD_SEC 
  });

  useEffect(() => {
    fetch(`${AUDIO_BACKEND_BASE}/training/config`)
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data) setConfig(data);
      })
      .catch(() => {
        // Fallback to default if fetch fails
      });
  }, []);

  return config;
}

// ---------------------------------------------------------------------------
// Training-status hook — reduced polling; WS pushes cover most updates
// ---------------------------------------------------------------------------
function useTrainingStatus(onProfileTab, trainingRunning) {
  const [data, setData] = useState(null);
  const [failures, setFailures] = useState(0);
  const timerRef = useRef(null);

  const poll = useCallback(async () => {
    try {
      const r = await fetch(`${AUDIO_BACKEND_BASE}/training/status`);
      if (!r.ok) throw new Error();
      const data = await r.json();
      console.log('[useTrainingStatus] Fetched data:', data);
      setData(data);
      setFailures(0);
    } catch {
      setFailures(n => n + 1);
    }
  }, []);

  // Initial fetch on mount
  useEffect(() => {
    poll();
  }, [poll]);

  useEffect(() => {
    clearInterval(timerRef.current);
    // Only poll when the Profile tab is open or training is actively running.
    // Much less aggressive than the main-status polling was.
    const interval = onProfileTab
      ? (trainingRunning ? 10_000 : 30_000)
      : trainingRunning ? 15_000
      : null;
    if (!interval) return;
    timerRef.current = setInterval(poll, interval);
    return () => clearInterval(timerRef.current);
  }, [onProfileTab, trainingRunning, poll]);

  useEffect(() => {
    const fn = () => { if (document.visibilityState === 'visible') poll(); };
    document.addEventListener('visibilitychange', fn);
    return () => document.removeEventListener('visibilitychange', fn);
  }, [poll]);

  return { data, failures };
}

function useShortcuts(activeTab, userPrompt) {
  const [data, setData] = useState(null);
  const refetch = useCallback(async () => {
    try {
      const r = await fetch(`${AUDIO_BACKEND_BASE}/shortcuts`);
      if (r.ok) setData(await r.json());
    } catch { /* silent */ }
  }, []);

  useEffect(() => {
    refetch();
  }, [refetch]);

  useEffect(() => {
    if (activeTab !== 'MAIN') return;
    const interval = setInterval(refetch, 8000);
    return () => clearInterval(interval);
  }, [activeTab, refetch]);

  useEffect(() => {
    if (userPrompt && typeof userPrompt === 'string' && userPrompt.includes('saved.')) {
      refetch();
    }
  }, [userPrompt, refetch]);

  useEffect(() => {
    const fn = () => { if (document.visibilityState === 'visible') refetch(); };
    document.addEventListener('visibilitychange', fn);
    return () => document.removeEventListener('visibilitychange', fn);
  }, [refetch]);

  return { shortcuts: data?.shortcuts ?? {}, refetch };
}

// ---------------------------------------------------------------------------
// Small shared components
// ---------------------------------------------------------------------------
function PillToggle({ id, checked, onChange, disabled = false }) {
  return (
    <button
      id={id}
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      className={`pill${checked ? ' pill--on' : ''}${disabled ? ' pill--disabled' : ''}`}
      onClick={() => !disabled && onChange(!checked)}
    >
      <span className="pill__thumb" aria-hidden="true" />
      <span className="sr-only">{checked ? 'On' : 'Off'}</span>
    </button>
  );
}

function PrefRow({ label, hint, id, checked, onChange, disabled }) {
  return (
    <div className="pref-row">
      <label className="pref-row__label" htmlFor={id}>
        <span className="pref-row__name">{label}</span>
        {hint && <span className="pref-row__hint">{hint}</span>}
      </label>
      <PillToggle id={id} checked={checked} onChange={onChange} disabled={disabled} />
    </div>
  );
}

function ConfirmModal({ title, body, confirmLabel = 'Confirm', onConfirm, onCancel }) {
  const firstBtnRef = useRef(null);

  useEffect(() => {
    firstBtnRef.current?.focus();
    const onKey = (e) => { if (e.key === 'Escape') onCancel(); };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [onCancel]);

  return (
    <div className="modal-veil" onClick={onCancel}>
      <div
        className="modal-box"
        role="dialog"
        aria-modal="true"
        aria-labelledby="modal-title"
        onClick={e => e.stopPropagation()}
      >
        <h2 id="modal-title" className="modal-box__title">{title}</h2>
        <p className="modal-box__body">{body}</p>
        <div className="modal-box__actions">
          <button
            ref={firstBtnRef}
            type="button"
            className="modal-box__btn modal-box__btn--confirm"
            onClick={onConfirm}
          >
            {confirmLabel}
          </button>
          <button
            type="button"
            className="modal-box__btn modal-box__btn--cancel"
            onClick={onCancel}
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Profile tab
// ---------------------------------------------------------------------------
function ProfileTab({ prefs, savePrefs, trainingStatus, pollFailures, addToast, trainingThreshold }) {
  const [showLaunchModal, setShowLaunchModal] = useState(false);
  const [showCancelModal, setShowCancelModal] = useState(false);
  const [showDataWarningModal, setShowDataWarningModal] = useState(false);
  const [launching, setLaunching] = useState(false);
  const [cancelling, setCancelling] = useState(false);

  if (!prefs) {
    return <div className="profile-tab"><p className="profile-tab__loading">Loading…</p></div>;
  }

  const accSec     = trainingStatus?.accumulated_audio_seconds ?? 0;
  const pct        = Math.min(100, Math.round((accSec / trainingThreshold) * 100));
  const inProgress = trainingStatus?.training_in_progress ?? false;
  const completed  = trainingStatus?.training_completed   ?? false;
  const error      = trainingStatus?.training_error ?? null;
  const canLaunch  = accSec >= trainingThreshold && !inProgress && !completed;
  
  // Display in seconds if threshold < 60, otherwise minutes with 1 decimal
  const useSeconds = trainingThreshold < 60;
  const displayValue = useSeconds ? accSec : (accSec / 60).toFixed(1);
  const displayThreshold = useSeconds ? trainingThreshold : (trainingThreshold / 60).toFixed(1);
  const displayUnit = useSeconds ? 'sec' : 'min';
  
  const stale      = pollFailures >= 3;

  // Debug logging
  console.log('[ProfileTab] Debug:', {
    trainingStatus,
    accSec,
    displayValue,
    threshold: trainingThreshold,
    displayThreshold,
    displayUnit,
    pct
  });

  const handleLaunch = async () => {
    setLaunching(true);
    setShowLaunchModal(false);
    try {
      const r = await fetch(`${AUDIO_BACKEND_BASE}/training/start`, { method: 'POST' });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        addToast(err.detail ?? 'Could not start training. Please try again.');
      } else {
        addToast('Training started in background');
      }
    } catch {
      addToast('Server unavailable — could not start training.');
    } finally {
      setLaunching(false);
    }
  };

  const handleCancel = async () => {
    setCancelling(true);
    setShowCancelModal(false);
    try {
      const r = await fetch(`${AUDIO_BACKEND_BASE}/training/cancel`, { method: 'POST' });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        addToast(err.detail ?? 'Could not cancel training.');
      } else {
        addToast('Training cancelled');
      }
    } catch {
      addToast('Server unavailable — could not cancel training.');
    } finally {
      setCancelling(false);
    }
  };

  const handleTrainingToggle = (enabled) => {
    if (enabled && !prefs.custom_training_enabled) {
      // Show warning when enabling for the first time
      setShowDataWarningModal(true);
    } else {
      // Directly disable without warning
      savePrefs({ custom_training_enabled: enabled });
    }
  };

  const confirmDataCollection = () => {
    setShowDataWarningModal(false);
    savePrefs({ custom_training_enabled: true });
  };

  return (
    <div className="profile-tab">
      <section className="pt-section">
        <h2 className="pt-section__title">Preferences</h2>
        <div className="pref-card">
          <PrefRow
            id="pref-guardrails"
            label="Guardrails"
            hint="Block potentially harmful commands"
            checked={prefs.guardrails_enabled ?? true}
            onChange={v => savePrefs({ guardrails_enabled: v })}
          />
          <PrefRow
            id="pref-training"
            label="Collect Training Data"
            hint="Record audio to personalise your model"
            checked={prefs.custom_training_enabled ?? false}
            onChange={handleTrainingToggle}
          />
          <PrefRow
            id="pref-shortcuts"
            label="Custom Shortcuts"
            hint="Use your saved phrase→command mappings"
            checked={prefs.custom_shortcuts_enabled ?? false}
            onChange={v => savePrefs({ custom_shortcuts_enabled: v })}
          />
        </div>
      </section>

      <section className="pt-section">
        <h2 className="pt-section__title">Voice Model Training</h2>
        <div className="collect-card">
          <div className="status-card">
            <span className="status-card__label">Audio collected</span>
            <span className="status-card__value">{displayValue} / {displayThreshold} {displayUnit}</span>
          </div>

          <div
            className="progress-track"
            role="progressbar"
            aria-valuenow={pct}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label={`Training data: ${pct}% complete`}
          >
            <div
              className={`progress-track__fill${completed ? ' progress-track__fill--done' : ''}`}
              style={{ width: `${pct}%` }}
            />
          </div>

          {stale && (
            <p className="stale-warning" role="alert">
              ⚠ Status may be out of date — server not responding.
            </p>
          )}

          {error && (
            <p className="error-warning" role="alert">
              ⚠ Training error: {error}
            </p>
          )}

          <div className="launch-block">
            {completed ? (
              <p className="launch-block__done">
                ✓ Training complete. Restart the server to activate your model.
              </p>
            ) : inProgress ? (
              <>
                <p className="launch-block__running" aria-live="polite">
                  <span className="launch-block__pulse" aria-hidden="true" />
                  Training in progress…
                </p>
                <button
                  type="button"
                  className="launch-block__btn launch-block__btn--cancel"
                  disabled={cancelling}
                  onClick={() => setShowCancelModal(true)}
                >
                  {cancelling ? 'Cancelling…' : 'Cancel Training'}
                </button>
              </>
            ) : (
              <button
                type="button"
                className="launch-block__btn"
                disabled={!canLaunch || launching}
                aria-disabled={!canLaunch || launching}
                onClick={() => setShowLaunchModal(true)}
              >
                {launching ? 'Starting…' : 'Launch Customisation'}
              </button>
            )}
            {!canLaunch && !inProgress && !completed && (
              <p className="launch-block__hint">
                Collect {displayThreshold} {displayUnit} of audio to unlock.
              </p>
            )}
          </div>
        </div>
      </section>

      {showLaunchModal && (
        <ConfirmModal
          title="Start voice model training?"
          body="Training runs in the background. The server will remain responsive, but training may take several minutes depending on your data size."
          confirmLabel="Start Training"
          onConfirm={handleLaunch}
          onCancel={() => setShowLaunchModal(false)}
        />
      )}

      {showCancelModal && (
        <ConfirmModal
          title="Cancel training?"
          body="This will stop the current training job. Progress will not be saved."
          confirmLabel="Cancel Training"
          onConfirm={handleCancel}
          onCancel={() => setShowCancelModal(false)}
        />
      )}

      {showDataWarningModal && (
        <ConfirmModal
          title="Enable Training Data Collection?"
          body="This will store audio recordings locally on the server to personalize your voice model. For 30 minutes of audio, approximately 150-200 MB of storage will be used. Audio files are saved in WAV format."
          confirmLabel="Enable Data Collection"
          onConfirm={confirmDataCollection}
          onCancel={() => setShowDataWarningModal(false)}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------
function App() {
  // Confirmation state
  const [awaitingConfirmation, setAwaitingConfirmation] = useState(false);
  const [lastCommandId, setLastCommandId] = useState(null);
  const [pendingCommand, setPendingCommand] = useState(null);
  const [cancelled, setCancelled] = useState(false);

  const [isListening, setIsListening] = useState(true);
  const [status, setStatus] = useState('listening');
  const [error, setError] = useState(null);
  const [isLightMode, setIsLightMode] = useState(false);
  const [feedbackItems, setFeedbackItems] = useState([]);
  const [userPrompt, setUserPrompt] = useState(null);
  const [userTranscript, setUserTranscript] = useState(null);

  const errorFeedbackRef = useRef(null);
  const uiClientRef = useRef(null);
  const auditLoggerRef = useRef(null);
  const accessibilityLayerRef = useRef(null);

  const [showThumbs, setShowThumbs] = useState(true);

  const [activeTab, setActiveTab] = useState('MAIN');
  const [toasts, setToasts] = useState([]);
  const [bannerDismissed, setBannerDismissed] = useState(false);
  const prevTrainingRunningRef = useRef(false);
  
  // Track shown notifications to prevent duplicates
  const shownNotifications = useRef({
    trainingComplete: false,
    lastEpoch: -1,
  });

  // Refs used inside the WS message handler (stale-closure-safe)
  const cancelledRef = useRef(false);
  const cancelledTranscriptRef = useRef(null);
  const confirmedRef = useRef(false);

  useEffect(() => { cancelledRef.current = cancelled; }, [cancelled]);

  const addToast = useCallback((message) => {
    setToasts(prev => [...prev, { id: Date.now() + Math.random(), message }]);
  }, []);

  const { prefs, save: savePrefs } = usePreferences(addToast);

  const trainingConfig = useTrainingConfig();

  const { data: trainingStatus, failures: pollFailures } = useTrainingStatus(
    activeTab === 'PROFILE',
    prefs?.training_in_progress ?? false,
  );

  const { shortcuts, refetch: refetchShortcuts } = useShortcuts(activeTab, userPrompt);

  // "model ready"  on training completion
  // "model ready" toast on training completion
  useEffect(() => {
    if (prevTrainingRunningRef.current && trainingStatus?.training_completed) {
      // Only show if we haven't already shown the completion notification
      if (!shownNotifications.current.trainingComplete) {
        addToast('Your voice model is ready. Restart the server to activate it.');
        setBannerDismissed(false);
        shownNotifications.current.trainingComplete = true;
      }
    }
    
    // Reset the flag when a new training starts
    if (trainingStatus?.training_in_progress) {
      shownNotifications.current.trainingComplete = false;
      shownNotifications.current.lastEpoch = -1;
    }
    
    prevTrainingRunningRef.current = trainingStatus?.training_in_progress ?? false;
  }, [trainingStatus?.training_in_progress, trainingStatus?.training_completed, addToast]);

  // -------------------------------------------------------------------------
  // Central WebSocket message handler — replaces the 400 ms setInterval loop
  // -------------------------------------------------------------------------
  const CANCEL_WORDS = new Set(['no', 'no.', 'cancel', 'nope', 'nope.', 'cancel that']);

  const handleServerMessage = useCallback((data) => {
    // Ignore keepalive frames
    if (data.type === 'ping' || data.type === 'pong') return;

    // --- Training events ---
    if (data.type === 'training_progress') {
      // Real-time epoch progress - only show each epoch once
      const epoch = data.epoch ?? -1;
      if (epoch !== shownNotifications.current.lastEpoch) {
        const loss = data.avg_loss != null ? data.avg_loss.toFixed(4) : 'N/A';
        addToast(`Training: Epoch ${epoch} completed (loss: ${loss})`);
        shownNotifications.current.lastEpoch = epoch;
      }
      return;
    }

    if (data.type === 'training_complete') {
      // Only show completion notification once
      if (!shownNotifications.current.trainingComplete) {
        const evalLoss = data.eval_loss != null ? data.eval_loss.toFixed(4) : 'N/A';
        addToast(`Training completed! Eval loss: ${evalLoss}`);
        shownNotifications.current.trainingComplete = true;
      }
      // Force reload training status after a brief delay
      setTimeout(() => {
        fetch(`${AUDIO_BACKEND_BASE}/training/status`)
          .then(r => r.ok ? r.json() : null)
          .catch(() => {});
      }, 500);
      return;
    }

    if (data.type === 'training_error') {
      addToast(`Training failed: ${data.error ?? 'Unknown error'}`);
      // Reset flags on error
      shownNotifications.current.trainingComplete = false;
      shownNotifications.current.lastEpoch = -1;
      return;
    }

    const newAwaiting    = data.awaiting_confirmation ?? false;
    const newPending     = data.pending_command ?? null;
    const newTranscript  = data.user_transcript ?? null;
    const lastAction     = data.last_action ?? null;
    const captureState   = data.capture_state ?? null;

    // --- last_action handling ---
    if (lastAction === 'cancelled') {
      setCancelled(true);
      cancelledRef.current = true;
      cancelledTranscriptRef.current = newTranscript;
    } else if (lastAction === 'confirmed') {
      confirmedRef.current = true;
      setCancelled(false);
      cancelledRef.current = false;
    }

    setAwaitingConfirmation(newAwaiting);
    setPendingCommand(newPending);
    setLastCommandId(data.last_command ?? null);

    // --- transcript / prompt update ---
    const isNewRealTranscript =
      newTranscript &&
      newTranscript.trim() !== '' &&
      newTranscript !== cancelledTranscriptRef.current &&
      !CANCEL_WORDS.has(newTranscript.trim().toLowerCase());

    if (isNewRealTranscript) {
      setCancelled(false);
      cancelledRef.current = false;
      confirmedRef.current = false;
      cancelledTranscriptRef.current = null;
      setUserPrompt(data.user_prompt ?? null);
      setUserTranscript(newTranscript);
    } else if (!cancelledRef.current) {
      setUserPrompt(data.user_prompt ?? null);
      setUserTranscript(newTranscript);
    }

    // --- audio capture state → UI status ---
    if (captureState) {
      if (captureState === 'speech_detected' || captureState === 'recording') {
        setStatus('recording');
      } else if (captureState === 'processing') {
        setStatus('processing');
      } else {
        setStatus('listening');
      }
    }

    // --- top-level state ---
    if (data.state) {
      if (data.state === 'idle') setStatus('idle');
      else if (data.state === 'processing') setStatus('processing');
      else if (data.state === 'executing') setStatus('processing');
      else if (data.state === 'listening' && !captureState) setStatus('listening');
    }

    // --- errors ---
    if (data.error) setError(data.error);
    else if (data.type !== 'error') setError(null);
  }, [addToast]); // eslint-disable-line react-hooks/exhaustive-deps

  useServerSocket(handleServerMessage);

  // -------------------------------------------------------------------------
  // Voice mode start/stop — still uses REST (fire-and-forget)
  // -------------------------------------------------------------------------
  useEffect(() => {
    const base = AUDIO_BACKEND_BASE;
    if (isListening) {
      fetch(`${base}/audio/capture/start`, { method: 'POST' })
        .then(r => r.json())
        .then(data => { if (!data.ok) console.warn('Backend capture start:', data.message); })
        .catch(err => console.warn('Failed to start backend capture:', err));
    } else {
      fetch(`${base}/audio/capture/stop`, { method: 'POST' }).catch(() => {});
    }
  }, [isListening]);

  // -------------------------------------------------------------------------
  // Send UI feedback (thumbs)
  // -------------------------------------------------------------------------
  const sendThumbsFeedback = (value) => {
    fetch(`${AUDIO_BACKEND_BASE}/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        command_id: lastCommandId || pendingCommand || 'unknown',
        feedback_type: 'ui',
        value,
        source: 'ui',
        command_text: pendingCommand || userPrompt || ''
      })
    }).then(() => {
      setAwaitingConfirmation(false);
      setPendingCommand(null);
      if (value === 'thumbs_down') {
        setCancelled(true);
        cancelledTranscriptRef.current = userTranscript;
      } else {
        setCancelled(false);
        confirmedRef.current = true;
      }
    }).catch(() => {});
  };

  // -------------------------------------------------------------------------
  // Confirmation UI
  // -------------------------------------------------------------------------
  const renderConfirmationUI = () => {
    if (!awaitingConfirmation) return null;
    return (
      <section
        className="confirmation-ui"
        aria-label="Awaiting verbal confirmation"
        aria-live="polite"
      >
        <div className="confirmation-verbal-cues">
          <span className="confirmation-cue confirmation-cue--yes" aria-label="Say Yes to confirm">
            🎙 Say <strong>"Yes"</strong> to confirm
          </span>
          <span className="confirmation-cue confirmation-cue--no" aria-label="Say No to cancel">
            🎙 Say <strong>"No"</strong> to cancel
          </span>
        </div>

        <div className="confirmation-thumbs-row">
          <button
            type="button"
            className="confirmation-thumbs-toggle"
            onClick={() => setShowThumbs(prev => !prev)}
            aria-label={showThumbs ? 'Hide button controls' : 'Show button controls'}
            title={showThumbs ? 'Hide buttons' : 'Show buttons'}
          >
            {showThumbs ? 'Hide buttons ▲' : 'Show buttons ▼'}
          </button>

          {showThumbs && (
            <div className="confirmation-thumbs-buttons" role="group" aria-label="Button alternatives">
              <button
                type="button"
                className="confirmation-btn confirmation-btn--up"
                onClick={() => sendThumbsFeedback('thumbs_up')}
                aria-label="Thumbs up — confirm command"
              >
                👍 Confirm
              </button>
              <button
                type="button"
                className="confirmation-btn confirmation-btn--down"
                onClick={() => sendThumbsFeedback('thumbs_down')}
                aria-label="Thumbs down — cancel command"
              >
                👎 Cancel
              </button>
            </div>
          )}
        </div>
      </section>
    );
  };

  // -------------------------------------------------------------------------
  // Initialise utility modules once
  // -------------------------------------------------------------------------
  useEffect(() => {
    auditLoggerRef.current = new AuditLogger();
    accessibilityLayerRef.current = new AccessibilityLayer();
    uiClientRef.current = new UiClient(setFeedbackItems, setError);
    errorFeedbackRef.current = new ErrorFeedback(
      uiClientRef.current,
      auditLoggerRef.current,
      accessibilityLayerRef.current
    );
    return () => { errorFeedbackRef.current?.clearAll(); };
  }, []);

  // -------------------------------------------------------------------------
  // Keyboard shortcuts
  // -------------------------------------------------------------------------
  useEffect(() => {
    const handleKeyDown = (e) => {
      const root = document.getElementById('voicebridge-root');
      if (!root || root.style.display === 'none') return;
      if (e.altKey && (e.key === 'l' || e.key === 'L')) { setIsLightMode(prev => !prev); e.preventDefault(); }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  // -------------------------------------------------------------------------
  // Error handlers
  // -------------------------------------------------------------------------
  const handleAudioData = async () => {
    // Server-side capture; status arrives via WebSocket
  };

  const handleError = (errorMessage) => {
    let errorCode = 'GENERIC_ERROR';
    if (errorMessage.includes('microphone') || errorMessage.includes('Microphone')) {
      errorCode = 'MIC_PERMISSION_DENIED';
    } else if (errorMessage.includes('device') || errorMessage.includes('in use')) {
      errorCode = 'AUDIO_DEVICE_ERROR';
    }
    if (errorFeedbackRef.current) {
      errorFeedbackRef.current.showError(errorCode, errorMessage);
    } else {
      setError(errorMessage);
    }
    setStatus('idle');
    setIsListening(false);
  };

  const handleDismissFeedback = (feedbackId) => {
    errorFeedbackRef.current?.dismiss(feedbackId);
  };

  const handleRecoverySelect = (feedbackId, optionId) => {
    const item = feedbackItems.find(f => f.id === feedbackId);
    if (!item?.recoveryOptions) return;
    const option = item.recoveryOptions.options.find(o => o.id === optionId);
    if (!option) return;
    if (option.label.includes('Retry')) { setIsListening(true); setStatus('listening'); }
    else if (option.label.includes('Reload')) { window.location.reload(); }
    else if (option.label.includes('Settings')) { window.open('chrome://settings/content/microphone', '_blank'); }
    handleDismissFeedback(feedbackId);
  };

  const toggleTheme = () => setIsLightMode(prev => !prev);

  const isTrainingInProgress = trainingStatus?.training_in_progress ?? false;
  const isTrainingCompleted  = trainingStatus?.training_completed   ?? false;
  const showBanner = (isTrainingInProgress || isTrainingCompleted) && !bannerDismissed;

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------
  return (
    <div
      className={`App extension-overlay ${isLightMode ? 'light-mode' : 'dark-mode'}`}
      role="application"
      aria-label="VoiceBridge"
    >
      <div className="extension-container" tabIndex={0}>
        <header className="App-header">
          <div className="header-content">
            <div>
              <h1>
                <button
                  type="button"
                  className="title-btn"
                  onClick={() => setActiveTab('MAIN')}
                  aria-label="Go to VoiceBridge main page"
                  title="VoiceBridge home"
                >
                  VoiceBridge
                </button>
              </h1>
              <p className="subtitle">An accessibility interface</p>
            </div>
            <div className="header-controls">
              <button
                type="button"
                className="theme-toggle"
                onClick={toggleTheme}
                aria-label={isLightMode ? 'Switch to dark mode (Alt+L)' : 'Switch to light mode (Alt+L)'}
                title={isLightMode ? 'Dark mode (Alt+L)' : 'Light mode (Alt+L)'}
              >
                {isLightMode ? '🌙' : '☀️'}
              </button>
              <button
                type="button"
                className={`profile-icon-btn${activeTab === 'PROFILE' ? ' profile-icon-btn--active' : ''}`}
                onClick={() => setActiveTab(activeTab === 'PROFILE' ? 'MAIN' : 'PROFILE')}
                aria-label="Profile & Training"
                title="Profile & Training"
              >
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="20" height="20" aria-hidden="true">
                  <path d="M12 12c2.7 0 4.8-2.1 4.8-4.8S14.7 2.4 12 2.4 7.2 4.5 7.2 7.2 9.3 12 12 12zm0 2.4c-3.2 0-9.6 1.6-9.6 4.8v2.4h19.2v-2.4c0-3.2-6.4-4.8-9.6-4.8z"/>
                </svg>
              </button>
            </div>
          </div>
        </header>

        <main className="App-main">
          {activeTab === 'MAIN' && showBanner && (
            <>
              {isTrainingInProgress && (
                <div className="train-banner train-banner--running" role="status" aria-live="polite">
                  <span className="train-banner__indicator train-banner__indicator--running" aria-hidden="true" />
                  <span className="train-banner__text">
                    Training in progress - VoiceBridge still responsive
                  </span>
                  <button
                    type="button"
                    className="train-banner__dismiss"
                    onClick={() => setBannerDismissed(true)}
                    aria-label="Dismiss notification"
                  >
                    ✕
                  </button>
                </div>
              )}
              {isTrainingCompleted && (
                <div className="train-banner train-banner--done" role="status" aria-live="polite">
                  <span className="train-banner__indicator train-banner__indicator--done" aria-hidden="true" />
                  <span className="train-banner__text">
                    Voice model ready — restart the server to activate.
                  </span>
                  <button
                    type="button"
                    className="train-banner__dismiss"
                    onClick={() => setBannerDismissed(true)}
                    aria-label="Dismiss notification"
                  >
                    ✕
                  </button>
                </div>
              )}
            </>
          )}

          <div className="main-container">
            {activeTab === 'PROFILE' ? (
              <ProfileTab
                prefs={prefs}
                savePrefs={savePrefs}
                trainingStatus={trainingStatus}
                pollFailures={pollFailures}
                addToast={addToast}
                trainingThreshold={trainingConfig.audio_threshold_seconds}
              />
            ) : (
              <>
                <div className="left-panel">
                  <StatusIndicator status={status} error={error} isLightMode={isLightMode} />
                  {(status === 'listening' || awaitingConfirmation) && (
                    <SpeechInput
                      isListening={isListening}
                      status={status}
                      onAudioData={handleAudioData}
                      onError={handleError}
                      isLightMode={isLightMode}
                      backendBase={AUDIO_BACKEND_BASE}
                    />
                  )}
                  {renderConfirmationUI()}
                  {Object.keys(shortcuts).length > 0 && (
                    <section className="shortcuts-bar" aria-label="Shortcuts">
                      <h3 className="shortcuts-bar__title">Shortcuts</h3>
                      <div className="shortcuts-bar__list">
                        {Object.entries(shortcuts).map(([id, s]) => (
                          <button
                            key={id}
                            type="button"
                            className="shortcuts-bar__btn"
                            onClick={async () => {
                              try {
                                const r = await fetch(`${AUDIO_BACKEND_BASE}/shortcuts/${id}/run`, { method: 'POST' });
                                if (!r.ok) throw new Error();
                                refetchShortcuts();
                              } catch {
                                addToast('Could not run shortcut.');
                              }
                            }}
                          >
                            {s.name || `shortcut_${id}`}
                          </button>
                        ))}
                      </div>
                    </section>
                  )}
                </div>
                <div className="right-panel">
                  {userPrompt && !cancelled && (awaitingConfirmation || !pendingCommand) && (
                    <div
                      className={`system-message-alert${awaitingConfirmation ? ' confirmation-ready' : ''}`}
                      style={{ marginBottom: 20, maxWidth: '100%' }}
                    >
                      <div className="system-message-icon">
                        {awaitingConfirmation ? '✅' : 'ℹ️'}
                      </div>
                      <div className="system-message-content">
                        <strong>{awaitingConfirmation ? 'Ready to Execute:' : 'System Message:'}</strong>
                        <p>{userPrompt}</p>
                      </div>
                    </div>
                  )}
                  <section className="llm-response-panel" aria-label="User Transcript">
                    <h2 className="llm-response-heading">Transcribed Audio</h2>
                    <div className="llm-response-content" style={{ maxHeight: 'none', overflow: 'visible' }}>
                      <div className="transcript-text">
                        {cancelled
                          ? '❌ Cancelled'
                          : userTranscript && userTranscript.trim() !== ''
                            ? userTranscript
                            : userPrompt
                              ? '—'
                              : 'Say a command to proceed.'}
                      </div>
                    </div>
                  </section>
                </div>
              </>
            )}
          </div>
        </main>

        <FeedbackDisplay
          feedbackItems={feedbackItems}
          onDismiss={handleDismissFeedback}
          onRecoverySelect={handleRecoverySelect}
          isLightMode={isLightMode}
        />
      </div>

      {/* Toast stack */}
      <div className="toast-stack" aria-live="assertive" aria-atomic="false">
        {toasts.map(t => (
          <div key={t.id} className="toast" role="alert">
            <span className="toast__msg">{t.message}</span>
            <button
              className="toast__x"
              onClick={() => setToasts(prev => prev.filter(x => x.id !== t.id))}
              aria-label="Dismiss notification"
            >✕</button>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;