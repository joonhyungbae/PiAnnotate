import { useAppStore } from '../../stores/useAppStore';
import './SettingsPanel.css';

interface SettingsPanelProps {
  onCameraPresetChange: (preset: 'top' | 'front' | 'back' | 'left' | 'right') => void;
}

export function SettingsPanel({ onCameraPresetChange }: SettingsPanelProps) {
  const { settings, updateSettings } = useAppStore();

  return (
    <div className="settings-panel">
      <h3>Settings</h3>

      <div className="setting-item">
        <label>
          <input
            type="checkbox"
            checked={settings.showFingering}
            onChange={(e) => updateSettings({ showFingering: e.target.checked })}
          />
          Show Fingering
        </label>
      </div>

      <div className="setting-item">
        <label>
          <input
            type="checkbox"
            checked={settings.showHittingPoints}
            onChange={(e) => updateSettings({ showHittingPoints: e.target.checked })}
          />
          Show Hitting Points
        </label>
      </div>

      <div className="setting-item">
        <label>Camera Preset</label>
        <select
          value={settings.cameraPreset}
          onChange={(e) => {
            const preset = e.target.value as 'top' | 'front' | 'back' | 'left' | 'right';
            updateSettings({ cameraPreset: preset });
            onCameraPresetChange(preset);
          }}
        >
          <option value="top">Top</option>
          <option value="front">Front</option>
          <option value="back">Back</option>
          <option value="left">Left</option>
          <option value="right">Right</option>
        </select>
      </div>
    </div>
  );
}

