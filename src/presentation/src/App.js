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
const TRAINING_THRESHOLD_SEC = 1800; // 30 minutes

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

function useTrainingStatus(onProfileTab, trainingRunning) {
  const [data, setData] = useState(null);
  const [failures, setFailures] = useState(0);
  const timerRef = useRef(null);

  const poll = useCallback(async () => {
    try {
      const r = await fetch(`${AUDIO_BACKEND_BASE}/training/status`);
      if (!r.ok) throw new Error();
      setData(await r.json());
      setFailures(0);
    } catch {
      setFailures(n => n + 1);
    }
  }, []);

  useEffect(() => {
    clearInterval(timerRef.current);
    const interval = onProfileTab
      ? (trainingRunning ? 10_000 : 30_000)
      : trainingRunning ? 15_000
      : null;
    if (!interval) return;
    poll();
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


function ProfileTab({ prefs, savePrefs, trainingStatus, pollFailures, addToast }) {
  const [showLaunchModal, setShowLaunchModal] = useState(false);
  const [launching, setLaunching] = useState(false);

  if (!prefs) {
    return <div className="profile-tab"><p className="profile-tab__loading">Loading…</p></div>;
  }

  const accSec     = trainingStatus?.accumulated_audio_seconds ?? 0;
  const pct        = Math.min(100, Math.round((accSec / TRAINING_THRESHOLD_SEC) * 100));
  const inProgress = trainingStatus?.training_in_progress ?? false;
  const completed  = trainingStatus?.training_completed   ?? false;
  const canLaunch  = accSec >= TRAINING_THRESHOLD_SEC && !inProgress && !completed;
  const mins       = Math.floor(accSec / 60);
  const stale      = pollFailures >= 3;

  const handleLaunch = async () => {
    setLaunching(true);
    setShowLaunchModal(false);
    try {
      const r = await fetch(`${AUDIO_BACKEND_BASE}/training/start`, { method: 'POST' });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        addToast(err.detail ?? 'Could not start training. Please try again.');
      }
    } catch {
      addToast('Server unavailable — could not start training.');
    } finally {
      setLaunching(false);
    }
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
            onChange={v => savePrefs({ custom_training_enabled: v })}
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
            <span className="status-card__value">{mins} / {TRAINING_THRESHOLD_SEC / 60} min</span>
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

          <div className="launch-block">
            {completed ? (
              <p className="launch-block__done">
                ✓ Training complete. Restart the server to activate your model.
              </p>
            ) : inProgress ? (
              <p className="launch-block__running" aria-live="polite">
                <span className="launch-block__pulse" aria-hidden="true" />
                Training in progress…
              </p>
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
                Collect {TRAINING_THRESHOLD_SEC / 60} minutes of audio to unlock.
              </p>
            )}
          </div>
        </div>
      </section>

      {showLaunchModal && (
        <ConfirmModal
          title="Start voice model training?"
          body="Training runs in the background. Please keep the app open until the training job completes"
          confirmLabel="Start Training"
          onConfirm={handleLaunch}
          onCancel={() => setShowLaunchModal(false)}
        />
      )}
    </div>
  );
}



function App() {
  // Confirmation state
  const [awaitingConfirmation, setAwaitingConfirmation] = useState(false);
  const [lastCommandId, setLastCommandId] = useState(null);
  const [pendingCommand, setPendingCommand] = useState(null);
  const [cancelled, setCancelled] = useState(false);

  const [isListening, setIsListening] = useState(true);
  const [status, setStatus] = useState('listening'); // idle, listening
  const [error, setError] = useState(null);
  const [isLightMode, setIsLightMode] = useState(false);
  const [feedbackItems, setFeedbackItems] = useState([]);
  const [userPrompt, setUserPrompt] = useState(null); // System message
  const [userTranscript, setUserTranscript] = useState(null); // User transcript

  // Initialize ErrorFeedback system
  const errorFeedbackRef = useRef(null);
  const uiClientRef = useRef(null);
  const auditLoggerRef = useRef(null);
  const accessibilityLayerRef = useRef(null);

  // Toggle thumbs button visibility
  const [showThumbs, setShowThumbs] = useState(true);

  // Customisation state
  const [activeTab, setActiveTab] = useState('MAIN');
  const [toasts, setToasts] = useState([]);
  const [bannerDismissed, setBannerDismissed] = useState(false);
  const prevTrainingRunningRef = useRef(false);

  const prevAwaitingRef = useRef(false);
  const prevPendingRef = useRef(null);
  const confirmedRef = useRef(false);
  const cancelledTranscriptRef = useRef(null);
  const cancelledRef = useRef(false);
  useEffect(() => { cancelledRef.current = cancelled; }, [cancelled]);

  const addToast = useCallback((message) => {
    setToasts(prev => [...prev, { id: Date.now() + Math.random(), message }]);
  }, []);

  const { prefs, save: savePrefs } = usePreferences(addToast);

  const { data: trainingStatus, failures: pollFailures } = useTrainingStatus(
    activeTab === 'PROFILE',
    prefs?.training_in_progress ?? false,
  );

  // "model ready"  on training completion
  useEffect(() => {
    if (prevTrainingRunningRef.current && trainingStatus?.training_completed) {
      addToast('Your voice model is ready. Restart the server to activate it.');
      setBannerDismissed(false);
    }
    prevTrainingRunningRef.current = trainingStatus?.training_in_progress ?? false;
  }, [trainingStatus?.training_in_progress, trainingStatus?.training_completed, addToast]);

  useEffect(() => {
    const tick = () => {
      fetch(`${AUDIO_BACKEND_BASE}/status`)
        .then((r) => r.json())
        .then((data) => {
          const newAwaiting = data.awaiting_confirmation ?? false;
          const newPending = data.pending_command ?? null;
          const newTranscript = data.user_transcript ?? null;
          const lastAction = data.last_action ?? null; // "confirmed" | "cancelled" | null

          if (lastAction === 'cancelled') {
            setCancelled(true);
            cancelledTranscriptRef.current = newTranscript;
          } else if (lastAction === 'confirmed') {
            confirmedRef.current = true;
            setCancelled(false);
          }

          prevAwaitingRef.current = newAwaiting;
          prevPendingRef.current = newPending;

          setAwaitingConfirmation(newAwaiting);
          setPendingCommand(newPending);
          setLastCommandId(data.last_command ?? null);

          // Update transcript and prompt if not cancelled
          const CANCEL_WORDS = new Set(['no', 'no.', 'cancel', 'nope', 'nope.', 'cancel that']);
          const isNewRealTranscript =
            newTranscript &&
            newTranscript.trim() !== '' &&
            newTranscript !== cancelledTranscriptRef.current &&
            !CANCEL_WORDS.has(newTranscript.trim().toLowerCase());

          if (isNewRealTranscript) {
            // New command spoken
            setCancelled(false);
            confirmedRef.current = false;
            cancelledTranscriptRef.current = null;
            setUserPrompt(data.user_prompt ?? null);
            setUserTranscript(newTranscript);
          } else if (!cancelledRef.current) {
            // Normal update
            setUserPrompt(data.user_prompt ?? null);
            setUserTranscript(newTranscript);
          }
          // Keep cancelled state if no new transcript
        })
        .catch(() => {});
    };
    tick(); 
    const interval = setInterval(tick, 400);
    return () => clearInterval(interval);
  }, []);

  // Send UI feedback (thumbs)
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
        // Clear cancelled state on confirm
        setCancelled(false);
        confirmedRef.current = true;
      }
    }).catch(() => {});
  };

  // Confirmation, only rendered when a complete command is pending verbal yes/no
  const renderConfirmationUI = () => {
    if (!awaitingConfirmation) return null;
    return (
      <section
        className="confirmation-ui"
        aria-label="Awaiting verbal confirmation"
        aria-live="polite"
      >
        {/* Verbal instructions */}
        <div className="confirmation-verbal-cues">
          <span className="confirmation-cue confirmation-cue--yes" aria-label="Say Yes to confirm">
            🎙 Say <strong>"Yes"</strong> to confirm
          </span>
          <span className="confirmation-cue confirmation-cue--no" aria-label="Say No to cancel">
            🎙 Say <strong>"No"</strong> to cancel
          </span>
        </div>

        {/* Thumbs toggle + buttons */}
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

  useEffect(() => {
    // Initialize modules
    auditLoggerRef.current = new AuditLogger();
    accessibilityLayerRef.current = new AccessibilityLayer();
    uiClientRef.current = new UiClient(setFeedbackItems, setError);
    errorFeedbackRef.current = new ErrorFeedback(
      uiClientRef.current,
      auditLoggerRef.current,
      accessibilityLayerRef.current
    );

    return () => {
      // Cleanup
      if (errorFeedbackRef.current) {
        errorFeedbackRef.current.clearAll();
      }
    };
  }, []);

      // Tell backend to start/stop audio capture (server records and saves to Recordings/)
  useEffect(() => {
    const base = AUDIO_BACKEND_BASE;

    if (isListening) {
      fetch(`${base}/audio/capture/start`, { method: 'POST' })
        .then((r) => r.json())
        .then((data) => {
          if (!data.ok) {
            console.warn('Backend capture start:', data.message);
          }
        })
        .catch((err) => {
          console.warn('Failed to start backend capture:', err);
        });
    } else {
      fetch(`${base}/audio/capture/stop`, { method: 'POST' }).catch(() => {});
    }
  }, [isListening]);

  // Poll backend capture status
  useEffect(() => {
    if (!isListening) return;
    const base = AUDIO_BACKEND_BASE;
    const interval = setInterval(() => {
      fetch(`${base}/audio/capture/status`)
        .then((r) => r.json())
        .then((data) => {
          if (data.capturing && data.state) {
            if (data.state === 'speech_detected' || data.state === 'recording') {
              setStatus('recording');
            } else if (data.state === 'processing') {
              setStatus('processing');
            } else {
              setStatus('listening');
            }
          }
        })
        .catch(() => {});
    }, 400);
    return () => clearInterval(interval);
  }, [isListening]);

  // Keyboard: light mode toggle (Alt+L) and collapse (Escape)
  useEffect(() => {
    const handleKeyDown = (e) => {
      const root = document.getElementById('voicebridge-root');
      const isWidgetVisible = root && root.style.display !== 'none';

      if (!isWidgetVisible) return;

      // Collapse widget: Escape
      if (e.key === 'Escape') {
        root.style.display = 'none';
        e.preventDefault();
        return;
      }
      // Toggle light mode: Alt+L (L for light)
      if (e.altKey && (e.key === 'l' || e.key === 'L')) {
        setIsLightMode(prev => !prev);
        e.preventDefault();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, []);

  const handleAudioData = async () => {
    // Server (src/server.py) captures and saves to Recordings/; status kept by polling /audio/capture/status
  };

  const handleError = (errorMessage) => {
    // Map common error messages to error codes
    let errorCode = 'GENERIC_ERROR';
    if (errorMessage.includes('microphone') || errorMessage.includes('Microphone')) {
      errorCode = 'MIC_PERMISSION_DENIED';
    } else if (errorMessage.includes('device') || errorMessage.includes('in use')) {
      errorCode = 'AUDIO_DEVICE_ERROR';
    }

    // Use ErrorFeedback to show error
    if (errorFeedbackRef.current) {
      errorFeedbackRef.current.showError(errorCode, errorMessage);
    } else {
      // Fallback to old error handling
      setError(errorMessage);
    }
    
    setStatus('idle');
    setIsListening(false);
  };

  const handleDismissFeedback = (feedbackId) => {
    if (errorFeedbackRef.current) {
      errorFeedbackRef.current.dismiss(feedbackId);
    }
  };

  const handleRecoverySelect = (feedbackId, optionId) => {
    const item = feedbackItems.find(f => f.id === feedbackId);
    if (!item || !item.recoveryOptions) return;

    const option = item.recoveryOptions.options.find(o => o.id === optionId);
    if (!option) return;

    // Handle recovery actions
    console.log('Recovery option selected:', option.label);
    
    // Example recovery actions
    if (option.label.includes('Retry')) {
      // Retry the failed operation
      setIsListening(true);
      setStatus('listening');
    } else if (option.label.includes('Reload')) {
      window.location.reload();
    } else if (option.label.includes('Settings')) {
      // Open settings (could navigate to chrome://settings/content/microphone)
      window.open('chrome://settings/content/microphone', '_blank');
    }

    // Dismiss the feedback after action
    handleDismissFeedback(feedbackId);
  };

  const toggleTheme = () => {
    setIsLightMode(!isLightMode);
  };

  const handleCollapse = () => {
    const root = document.getElementById('voicebridge-root');
    if (root) root.style.display = 'none';
  };

  const isTrainingInProgress = trainingStatus?.training_in_progress ?? false;
  const isTrainingCompleted  = trainingStatus?.training_completed   ?? false;
  const showBanner = (isTrainingInProgress || isTrainingCompleted) && !bannerDismissed;

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
              <h1>VoiceBridge</h1>
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
                className="minimize-button"
                onClick={handleCollapse}
                aria-label="Collapse widget (Escape)"
                title="Collapse (Escape)"
              >
                −
              </button>
            </div>
          </div>

          <nav className="tab-nav" aria-label="Dashboard sections">
            <button
              type="button"
              className={`tab-nav__btn${activeTab === 'MAIN' ? ' tab-nav__btn--active' : ''}`}
              onClick={() => setActiveTab('MAIN')}
              aria-current={activeTab === 'MAIN' ? 'page' : undefined}
            >
              VoiceBridge
            </button>
            <button
              type="button"
              className={`tab-nav__btn${activeTab === 'PROFILE' ? ' tab-nav__btn--active' : ''}`}
              onClick={() => setActiveTab('PROFILE')}
              aria-current={activeTab === 'PROFILE' ? 'page' : undefined}
            >
              Profile &amp; Training
              {isTrainingCompleted && (
                <span className="tab-nav__badge" aria-label="Restart required">↻</span>
              )}
            </button>
          </nav>
        </header>

        <main className="App-main">
          {activeTab === 'MAIN' && showBanner && (
            <div
              className={`train-banner${isTrainingCompleted ? ' train-banner--done' : ''}`}
              role="status"
              aria-live="polite"
            >
              <span
                className={`train-banner__indicator${isTrainingCompleted ? ' train-banner__indicator--done' : ' train-banner__indicator--pulse'}`}
                aria-hidden="true"
              />
              <span className="train-banner__text">
                {isTrainingCompleted
                  ? 'Voice model ready — restart the server to activate.'
                  : 'Training in progress…'}
              </span>
              {!isTrainingCompleted && (
                <button
                  type="button"
                  className="train-banner__dismiss"
                  onClick={() => setBannerDismissed(true)}
                  aria-label="Dismiss training banner"
                >✕</button>
              )}
            </div>
          )}

          <div className="main-container">
            {activeTab === 'PROFILE' ? (
              <ProfileTab
                prefs={prefs}
                savePrefs={savePrefs}
                trainingStatus={trainingStatus}
                pollFailures={pollFailures}
                addToast={addToast}
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
                </div>
                <div className="right-panel">
                  {userPrompt && !cancelled && (awaitingConfirmation || !pendingCommand) && (
                    <div
                      className={`system-message-alert${awaitingConfirmation ? ' confirmation-ready' : ''}`}
                      style={{marginBottom: 20, maxWidth: '100%'}}
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
                    <h2 className="llm-response-heading">Executing</h2>
                    <div className="llm-response-content" style={{maxHeight: 'none', overflow: 'visible'}}>
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