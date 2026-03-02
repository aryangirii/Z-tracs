import React from 'react';
import SummaryCards from '../components/SummaryCards';
import ForecastChart from '../components/ForecastChart';
import RiskPanel from '../components/RiskPanel';
import SimulationPanel from '../components/SimulationPanel';
import InfrastructurePanel from '../components/InfrastructurePanel';
import HealthStatus from '../components/HealthStatus';
import './Dashboard.css';

const Dashboard = () => {
  // placeholder data
  const summaryData = { average: 110, risk: 'Critical', peak: 91 };
  return (
    <div className="dashboard">
      <SummaryCards data={summaryData} />
      <ForecastChart />
      <RiskPanel risk="Critical" />
      <SimulationPanel />
      <InfrastructurePanel score={9.1} />
      <HealthStatus modelLoaded={true} version="1.0.0" />
    </div>
  );
};

export default Dashboard;
