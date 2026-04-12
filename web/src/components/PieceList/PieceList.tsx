import { useEffect, useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { fetchPiecesMetadata, fetchAnnotationStatus, fetchAnnotationSources, type AnnotationSources } from '../../api';
import type { PieceMetadata, SplitType, AnnotationStatus } from '../../types';
import './PieceList.css';

type SortKey = 'piece_id' | 'date' | 'name' | 'composer' | 'split' | 'dataset' | 'duration' | 'annotator' | 'status';
type SortDirection = 'asc' | 'desc';
type SplitFilter = 'all' | SplitType;
type StatusFilter = 'all' | 'r1_done' | 'r1_todo' | 'r2_done' | 'r2_todo' | 'r3_done' | 'r3_todo';
type AnnotatorFilter = 'all' | 1 | 2 | 3 | 4 | 5 | 6;

const SPLIT_ORDER: Record<string, number> = { train: 0, valid: 1, test: 2, unknown: 3 };
const FPS = 60;

// Convert num_frames to mm:ss format
function formatDuration(numFrames: number): string {
  const totalSeconds = Math.round(numFrames / FPS);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

// Format date as YYMMDD:HH:MM
function formatAnnotationDate(isoDateString: string | undefined): string {
  if (!isoDateString) return '-';
  try {
    const date = new Date(isoDateString);
    if (isNaN(date.getTime())) return '-';
    
    const yy = String(date.getFullYear()).slice(-2);
    const mm = String(date.getMonth() + 1).padStart(2, '0');
    const dd = String(date.getDate()).padStart(2, '0');
    const hh = String(date.getHours()).padStart(2, '0');
    const min = String(date.getMinutes()).padStart(2, '0');
    
    return `${yy}${mm}${dd}:${hh}:${min}`;
  } catch {
    return '-';
  }
}

// localStorage key for persisting filter state
const FILTER_STORAGE_KEY = 'pieceListFilters';

// Load saved filters from localStorage
function loadSavedFilters() {
  try {
    const saved = localStorage.getItem(FILTER_STORAGE_KEY);
    if (saved) {
      return JSON.parse(saved);
    }
  } catch (e) {
    console.warn('Failed to load saved filters:', e);
  }
  return null;
}

// Save filters to localStorage
function saveFilters(filters: {
  searchQuery: string;
  sortKey: SortKey;
  sortDirection: SortDirection;
  splitFilter: SplitFilter;
  statusFilter: StatusFilter;
  annotatorFilter: AnnotatorFilter;
}) {
  try {
    localStorage.setItem(FILTER_STORAGE_KEY, JSON.stringify(filters));
  } catch (e) {
    console.warn('Failed to save filters:', e);
  }
}

export function PieceList() {
  const navigate = useNavigate();
  const [pieces, setPieces] = useState<PieceMetadata[]>([]);
  const [annotationStatus, setAnnotationStatus] = useState<AnnotationStatus>({ review1: {}, review2: {}, review3: {} });
  const [annotationSources, setAnnotationSources] = useState<AnnotationSources>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Initialize state from localStorage or defaults
  const savedFilters = loadSavedFilters();
  const [searchQuery, setSearchQuery] = useState(savedFilters?.searchQuery || '');
  const [sortKey, setSortKey] = useState<SortKey>(savedFilters?.sortKey || 'piece_id');
  const [sortDirection, setSortDirection] = useState<SortDirection>(savedFilters?.sortDirection || 'asc');
  const [splitFilter, setSplitFilter] = useState<SplitFilter>(savedFilters?.splitFilter || 'all');
  const [statusFilter, setStatusFilter] = useState<StatusFilter>(savedFilters?.statusFilter || 'all');
  const [annotatorFilter, setAnnotatorFilter] = useState<AnnotatorFilter>(savedFilters?.annotatorFilter || 'all');
  const [showShortcuts, setShowShortcuts] = useState(false);
  
  // Save filters whenever they change
  useEffect(() => {
    saveFilters({
      searchQuery,
      sortKey,
      sortDirection,
      splitFilter,
      statusFilter,
      annotatorFilter,
    });
  }, [searchQuery, sortKey, sortDirection, splitFilter, statusFilter, annotatorFilter]);

  useEffect(() => {
    Promise.all([fetchPiecesMetadata(), fetchAnnotationStatus(), fetchAnnotationSources()])
      .then(([piecesData, statusData, sourcesData]) => {
        setPieces(piecesData);
        setAnnotationStatus(statusData);
        setAnnotationSources(sourcesData);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  // Check review 1 (annotator review) completion
  const isReview1Completed = (pieceId: number) => {
    return annotationStatus.review1?.[String(pieceId)] !== undefined;
  };

  // Check review 2 (pianist review) completion
  const isReview2Completed = (pieceId: number) => {
    return annotationStatus.review2?.[String(pieceId)] !== undefined;
  };

  // Check review 3 completion
  const isReview3Completed = (pieceId: number) => {
    return annotationStatus.review3?.[String(pieceId)] !== undefined;
  };

  // Get the most recent annotation date for a piece
  const getAnnotationDate = (pieceId: number): string | undefined => {
    const pieceIdStr = String(pieceId);
    const dates: string[] = [];
    
    // Collect all completed_at dates
    if (annotationStatus.review1?.[pieceIdStr]?.completed_at) {
      dates.push(annotationStatus.review1[pieceIdStr].completed_at);
    }
    if (annotationStatus.review2?.[pieceIdStr]?.completed_at) {
      dates.push(annotationStatus.review2[pieceIdStr].completed_at);
    }
    if (annotationStatus.review3?.[pieceIdStr]?.completed_at) {
      dates.push(annotationStatus.review3[pieceIdStr].completed_at);
    }
    if (annotationStatus.fingering_completed?.[pieceIdStr]?.completed_at) {
      dates.push(annotationStatus.fingering_completed[pieceIdStr].completed_at);
    }
    if (annotationStatus.post_completed?.[pieceIdStr]?.completed_at) {
      dates.push(annotationStatus.post_completed[pieceIdStr].completed_at);
    }
    
    if (dates.length === 0) return undefined;
    
    // Return the most recent date
    return dates.sort().reverse()[0];
  };

  // Get status score for sorting (0: both done, 1: r1 done only, 2: r2 done only, 3: none)
  const getStatusScore = (pieceId: number) => {
    const r1 = isReview1Completed(pieceId);
    const r2 = isReview2Completed(pieceId);
    if (r1 && r2) return 0;
    if (r1) return 1;
    if (r2) return 2;
    return 3;
  };

  const filteredAndSortedPieces = useMemo(() => {
    let result = [...pieces];

    // Split filter
    if (splitFilter !== 'all') {
      result = result.filter((piece) => piece.split === splitFilter);
    }

    // Status filter (3-stage)
    if (statusFilter !== 'all') {
      result = result.filter((piece) => {
        const r1Done = isReview1Completed(piece.piece_id);
        const r2Done = isReview2Completed(piece.piece_id);
        const r3Done = isReview3Completed(piece.piece_id);
        switch (statusFilter) {
          case 'r1_done': return r1Done;
          case 'r1_todo': return !r1Done;
          case 'r2_done': return r2Done;
          case 'r2_todo': return !r2Done;
          case 'r3_done': return r3Done;
          case 'r3_todo': return !r3Done;
          default: return true;
        }
      });
    }

    // Annotator filter
    if (annotatorFilter !== 'all') {
      result = result.filter((piece) => piece.annotator === annotatorFilter);
    }

    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      result = result.filter(
        (piece) =>
          piece.piece_id.toString().includes(query) ||
          piece.name.toLowerCase().includes(query) ||
          piece.composer.toLowerCase().includes(query)
      );
    }

    // Sort
    result.sort((a, b) => {
      let comparison = 0;
      if (sortKey === 'piece_id') {
        comparison = a.piece_id - b.piece_id;
      } else if (sortKey === 'date') {
        const dateA = getAnnotationDate(a.piece_id) || '';
        const dateB = getAnnotationDate(b.piece_id) || '';
        comparison = dateA.localeCompare(dateB);
      } else if (sortKey === 'split') {
        comparison = (SPLIT_ORDER[a.split || 'unknown'] || 3) - (SPLIT_ORDER[b.split || 'unknown'] || 3);
      } else if (sortKey === 'dataset') {
        comparison = (a.dataset || '').localeCompare(b.dataset || '');
      } else if (sortKey === 'duration') {
        comparison = (a.num_frames || 0) - (b.num_frames || 0);
      } else if (sortKey === 'annotator') {
        comparison = (a.annotator || 99) - (b.annotator || 99);
      } else if (sortKey === 'status') {
        comparison = getStatusScore(a.piece_id) - getStatusScore(b.piece_id);
      } else {
        comparison = (a[sortKey] || '').localeCompare(b[sortKey] || '');
      }
      return sortDirection === 'asc' ? comparison : -comparison;
    });

    return result;
  }, [pieces, searchQuery, sortKey, sortDirection, splitFilter, statusFilter, annotatorFilter, annotationStatus]);

  // Count pieces per split
  const splitCounts = useMemo(() => {
    return pieces.reduce(
      (acc, piece) => {
        const split = piece.split || 'unknown';
        acc[split] = (acc[split] || 0) + 1;
        return acc;
      },
      {} as Record<string, number>
    );
  }, [pieces]);

  // Count pieces per status (3-stage)
  const statusCounts = useMemo(() => {
    let r1Done = 0, r2Done = 0, r3Done = 0;
    pieces.forEach((piece) => {
      if (isReview1Completed(piece.piece_id)) r1Done++;
      if (isReview2Completed(piece.piece_id)) r2Done++;
      if (isReview3Completed(piece.piece_id)) r3Done++;
    });
    return {
      r1_done: r1Done,
      r1_todo: pieces.length - r1Done,
      r2_done: r2Done,
      r2_todo: pieces.length - r2Done,
      r3_done: r3Done,
      r3_todo: pieces.length - r3Done,
    };
  }, [pieces, annotationStatus]);

  // Count pieces per annotator
  const annotatorCounts = useMemo(() => {
    return pieces.reduce(
      (acc, piece) => {
        const annotator = piece.annotator;
        if (annotator) {
          acc[annotator] = (acc[annotator] || 0) + 1;
        }
        return acc;
      },
      {} as Record<number, number>
    );
  }, [pieces]);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDirection((prev) => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDirection('asc');
    }
  };

  const getSortIcon = (key: SortKey) => {
    if (sortKey !== key) return '↕';
    return sortDirection === 'asc' ? '↑' : '↓';
  };

  const handlePieceClick = (pieceId: number) => {
    navigate(`/vis?id=${pieceId}`);
  };

  if (loading) {
    return (
      <div className="piece-list-container">
        <div className="loading">
          <div className="loading-spinner"></div>
          <span>Loading pieces...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="piece-list-container">
        <div className="error">
          <i className="fas fa-exclamation-triangle" style={{ marginRight: 10 }}></i>
          Error: {error}
        </div>
      </div>
    );
  }

  return (
    <div className="piece-list-container">
      {/* AppBar */}
      <header className="appbar">
        {/* Search - Left aligned */}
        <div className="search-wrapper">
          <input
            type="text"
            className="search-input"
            placeholder="Search pieces..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          <i className="fas fa-search search-icon"></i>
          <button 
            className={`search-clear ${searchQuery ? 'visible' : ''}`}
            onClick={() => setSearchQuery('')}
            type="button"
          >
            <i className="fas fa-times"></i>
          </button>
        </div>

        {/* Controls - Right aligned */}
        <div className="appbar-controls">
          {/* Split filter buttons */}
          <div className="split-filter-group">
            <button
              className={`split-filter-btn ${splitFilter === 'all' ? 'active' : ''}`}
              onClick={() => setSplitFilter('all')}
            >
              All <span className="filter-count">{pieces.length}</span>
            </button>
            <button
              className={`split-filter-btn split-train ${splitFilter === 'train' ? 'active' : ''}`}
              onClick={() => setSplitFilter('train')}
            >
              Train <span className="filter-count">{splitCounts.train || 0}</span>
            </button>
            <button
              className={`split-filter-btn split-valid ${splitFilter === 'valid' ? 'active' : ''}`}
              onClick={() => setSplitFilter('valid')}
            >
              Valid <span className="filter-count">{splitCounts.valid || 0}</span>
            </button>
            <button
              className={`split-filter-btn split-test ${splitFilter === 'test' ? 'active' : ''}`}
              onClick={() => setSplitFilter('test')}
            >
              Test <span className="filter-count">{splitCounts.test || 0}</span>
            </button>
          </div>

          {/* Review 1 filter (Annotator) */}
          <div className="status-filter-group">
            <span className="filter-label">R1</span>
            <button
              className={`status-filter-btn ${statusFilter === 'all' ? 'active' : ''}`}
              onClick={() => setStatusFilter('all')}
            >
              All
            </button>
            <button
              className={`status-filter-btn status-completed ${statusFilter === 'r1_done' ? 'active' : ''}`}
              onClick={() => setStatusFilter('r1_done')}
            >
              ✓ <span className="filter-count">{statusCounts.r1_done}</span>
            </button>
            <button
              className={`status-filter-btn status-pending ${statusFilter === 'r1_todo' ? 'active' : ''}`}
              onClick={() => setStatusFilter('r1_todo')}
            >
              ○ <span className="filter-count">{statusCounts.r1_todo}</span>
            </button>
          </div>

          {/* Review 2 filter (Pianist) */}
          <div className="status-filter-group review2-filter">
            <span className="filter-label">R2</span>
            <button
              className={`status-filter-btn status-completed ${statusFilter === 'r2_done' ? 'active' : ''}`}
              onClick={() => setStatusFilter('r2_done')}
            >
              ✓ <span className="filter-count">{statusCounts.r2_done}</span>
            </button>
            <button
              className={`status-filter-btn status-pending ${statusFilter === 'r2_todo' ? 'active' : ''}`}
              onClick={() => setStatusFilter('r2_todo')}
            >
              ○ <span className="filter-count">{statusCounts.r2_todo}</span>
            </button>
          </div>

          {/* Review 3 filter */}
          <div className="status-filter-group review3-filter">
            <span className="filter-label">R3</span>
            <button
              className={`status-filter-btn status-completed ${statusFilter === 'r3_done' ? 'active' : ''}`}
              onClick={() => setStatusFilter('r3_done')}
            >
              ✓ <span className="filter-count">{statusCounts.r3_done}</span>
            </button>
            <button
              className={`status-filter-btn status-pending ${statusFilter === 'r3_todo' ? 'active' : ''}`}
              onClick={() => setStatusFilter('r3_todo')}
            >
              ○ <span className="filter-count">{statusCounts.r3_todo}</span>
            </button>
          </div>

          {/* Annotator filter buttons */}
          <div className="annotator-filter-group">
            <button
              className={`annotator-filter-btn ${annotatorFilter === 'all' ? 'active' : ''}`}
              onClick={() => setAnnotatorFilter('all')}
            >
              All
            </button>
            {[1, 2, 3, 4, 5, 6].map((id) => (
              <button
                key={id}
                className={`annotator-filter-btn annotator-${id} ${annotatorFilter === id ? 'active' : ''}`}
                onClick={() => setAnnotatorFilter(id as AnnotatorFilter)}
              >
                A{id} <span className="filter-count">{annotatorCounts[id] || 0}</span>
              </button>
            ))}
          </div>

          {/* Help Button */}
          <button
            className="help-btn"
            onClick={() => setShowShortcuts(!showShortcuts)}
            title="Keyboard Shortcuts"
          >
            <i className="fas fa-question-circle"></i>
          </button>
        </div>
      </header>

      {/* Table */}
      <div className="table-container">
        <table className="piece-table">
          <thead>
            <tr>
              <th onClick={() => handleSort('piece_id')} className="th-id">
                ID
                <span className="sort-icon">{getSortIcon('piece_id')}</span>
              </th>
              <th onClick={() => handleSort('date')} className="th-date">
                Date
                <span className="sort-icon">{getSortIcon('date')}</span>
              </th>
              <th onClick={() => handleSort('name')}>
                Piece Name
                <span className="sort-icon">{getSortIcon('name')}</span>
              </th>
              <th onClick={() => handleSort('composer')}>
                Composer
                <span className="sort-icon">{getSortIcon('composer')}</span>
              </th>
              <th onClick={() => handleSort('dataset')} className="th-dataset">
                Dataset
                <span className="sort-icon">{getSortIcon('dataset')}</span>
              </th>
              <th onClick={() => handleSort('split')} className="th-split">
                Split
                <span className="sort-icon">{getSortIcon('split')}</span>
              </th>
              <th onClick={() => handleSort('duration')} className="th-duration">
                Duration
                <span className="sort-icon">{getSortIcon('duration')}</span>
              </th>
              <th onClick={() => handleSort('annotator')} className="th-annotator">
                Annotator
                <span className="sort-icon">{getSortIcon('annotator')}</span>
              </th>
              <th onClick={() => handleSort('status')} className="th-status">
                Status
                <span className="sort-icon">{getSortIcon('status')}</span>
              </th>
              <th className="th-actions">
                Actions
              </th>
            </tr>
          </thead>
          <tbody>
            {filteredAndSortedPieces.map((piece, index) => (
              <tr key={piece.piece_id ?? index}>
                <td className="piece-id">{piece.piece_id}</td>
                <td className="piece-date">{formatAnnotationDate(getAnnotationDate(piece.piece_id))}</td>
                <td className="piece-name">{piece.name}</td>
                <td className="piece-composer">{piece.composer}</td>
                <td className="piece-dataset">
                  <span className="dataset-badge">{piece.dataset || '-'}</span>
                </td>
                <td className="piece-split">
                  <span className={`split-badge split-${piece.split || 'unknown'}`}>
                    {piece.split || 'unknown'}
                  </span>
                </td>
                <td className="piece-duration">
                  {piece.num_frames ? formatDuration(piece.num_frames) : '-'}
                </td>
                <td className="piece-annotator">
                  {piece.annotator ? (
                    <span className={`annotator-badge annotator-${piece.annotator}`}>
                      A{piece.annotator}
                    </span>
                  ) : (
                    <span className="annotator-badge annotator-none">-</span>
                  )}
                </td>
                <td className="piece-status">
                  <div className="status-badges">
                    {/* Annotation Source Badge */}
                    {annotationSources[String(piece.piece_id)] === 'r0' && (
                      <span 
                        className="annotation-source-badge source-r0"
                        title="AI Prior (r0)"
                      >
                        🤖
                      </span>
                    )}
                    {annotationSources[String(piece.piece_id)] === 'r1' && (
                      <span 
                        className="annotation-source-badge source-r1"
                        title="Human Annotation (r1)"
                      >
                        ✍️
                      </span>
                    )}
                    <span 
                      className={`review-badge review1 ${isReview1Completed(piece.piece_id) ? 'done' : 'todo'}`}
                      title="Review 1 (Annotator)"
                    >
                      R1
                    </span>
                    <span 
                      className={`review-badge review2 ${isReview2Completed(piece.piece_id) ? 'done' : 'todo'}`}
                      title="Review 2 (Pianist)"
                    >
                      R2
                    </span>
                    <span 
                      className={`review-badge review3 ${isReview3Completed(piece.piece_id) ? 'done' : 'todo'}`}
                      title="Review 3"
                    >
                      R3
                    </span>
                  </div>
                </td>
                <td className="piece-actions">
                  <div className="action-buttons">
                    <button 
                      className="action-btn annotate-btn"
                      onClick={(e) => {
                        e.stopPropagation();
                        handlePieceClick(piece.piece_id);
                      }}
                      title="Open annotation view"
                    >
                      Annotate
                    </button>
                    <button 
                      className="action-btn compare-btn"
                      onClick={(e) => {
                        e.stopPropagation();
                        navigate(`/fingering/${piece.piece_id}`);
                      }}
                      title="Compare fingerings"
                    >
                      Compare
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {filteredAndSortedPieces.length === 0 && (
        <div className="no-results">
          <div className="no-results-icon">
            <i className="fas fa-search"></i>
          </div>
          No pieces found matching "{searchQuery}"
        </div>
      )}

      {/* Help Popup */}
      {showShortcuts && (
        <div className="shortcuts-popup-overlay" onClick={() => setShowShortcuts(false)}>
          <div className="shortcuts-popup" onClick={(e) => e.stopPropagation()}>
            <div className="shortcuts-header">
              <h3>Help</h3>
              <button 
                className="close-shortcuts-btn"
                onClick={() => setShowShortcuts(false)}
              >
                <i className="fas fa-times"></i>
              </button>
            </div>
            <div className="shortcuts-content">
              <div className="shortcut-section">
                <h4>Keyboard Shortcuts</h4>
                <div className="shortcut-item">
                  <span className="shortcut-key">/</span>
                  <span className="shortcut-desc">Focus search</span>
                </div>
                <div className="shortcut-item">
                  <span className="shortcut-key">Esc</span>
                  <span className="shortcut-desc">Clear search</span>
                </div>
              </div>
              
              <div className="shortcut-section">
                <h4>Split Filter</h4>
                <div className="shortcut-item">
                  <span className="shortcut-key">All</span>
                  <span className="shortcut-desc">Show all pieces</span>
                </div>
                <div className="shortcut-item">
                  <span className="shortcut-key">Train</span>
                  <span className="shortcut-desc">Training dataset</span>
                </div>
                <div className="shortcut-item">
                  <span className="shortcut-key">Test</span>
                  <span className="shortcut-desc">Test dataset</span>
                </div>
              </div>
              
              <div className="shortcut-section">
                <h4>Status Filter</h4>
                <div className="shortcut-item">
                  <span className="shortcut-key">All</span>
                  <span className="shortcut-desc">All statuses</span>
                </div>
                <div className="shortcut-item">
                  <span className="shortcut-key">Todo</span>
                  <span className="shortcut-desc">R1 not completed</span>
                </div>
                <div className="shortcut-item">
                  <span className="shortcut-key">R1 ✓</span>
                  <span className="shortcut-desc">Review 1 completed</span>
                </div>
                <div className="shortcut-item">
                  <span className="shortcut-key">R2 ✓</span>
                  <span className="shortcut-desc">Review 2 completed</span>
                </div>
              </div>
              
              <div className="shortcut-section">
                <h4>Table</h4>
                <div className="shortcut-item">
                  <span className="shortcut-key">Click header</span>
                  <span className="shortcut-desc">Sort by column</span>
                </div>
                <div className="shortcut-item">
                  <span className="shortcut-key">Click row</span>
                  <span className="shortcut-desc">Open annotation page</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
