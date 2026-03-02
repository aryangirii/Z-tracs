import React from 'react';
import './SimulationPanel.css';

const SimulationPanel = ({ onSimulate }) => {
  return (
    <div className="simulation-panel">
      <h3>Scenario Simulation</h3>
      {/* controls would go here */}
      <button onClick={onSimulate}>Simulate</button>
    </div>
  );
};

export default SimulationPanel;
