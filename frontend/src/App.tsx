import { Routes, Route, NavLink } from 'react-router-dom';
import { useState } from 'react';
import Overview from './pages/Overview';
import Tasks from './pages/Tasks';
import Annotate from './pages/Annotate';
import Training from './pages/Training';
import Exports from './pages/Exports';
import CreateTaskModal from './components/CreateTaskModal';

function App() {
  const [showCreateTask, setShowCreateTask] = useState(false);

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div className="sidebar-logo">
          RL Platform <span>v0.1</span>
        </div>
        <nav>
          <NavLink to="/" end>Overview</NavLink>
          <NavLink to="/tasks">Tasks</NavLink>
          <NavLink to="/annotate">Annotate</NavLink>
          <NavLink to="/training">Training</NavLink>
          <NavLink to="/exports">Exports</NavLink>
        </nav>
        <div style={{ padding: '16px 20px' }}>
          <button className="btn btn-primary" style={{ width: '100%' }} onClick={() => setShowCreateTask(true)}>
            + New Task
          </button>
        </div>
      </aside>

      <main className="main-content">
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/tasks" element={<Tasks />} />
          <Route path="/annotate" element={<Annotate />} />
          <Route path="/training" element={<Training />} />
          <Route path="/exports" element={<Exports />} />
        </Routes>
      </main>

      {showCreateTask && <CreateTaskModal onClose={() => setShowCreateTask(false)} />}
    </div>
  );
}

export default App;
