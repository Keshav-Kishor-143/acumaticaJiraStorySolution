import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Paper,
  Typography,
  Chip,
  LinearProgress,
  Fade,
  useTheme,
  alpha,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  AutoAwesome as SparkleIcon,
  CheckCircle as CheckCircleIcon,
  RadioButtonUnchecked as RadioIcon,
  ContentCopy as ContentCopyIcon,
  Download as DownloadIcon,
} from '@mui/icons-material';
import toast from 'react-hot-toast';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/cjs/styles/prism';

const StreamingSolutionDisplay = ({ 
  solution, 
  storyId, 
  processingTime, 
  savedFilePath,
  isStreaming = false,
  progress = null 
}) => {
  const theme = useTheme();
  const [displayedSolution, setDisplayedSolution] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [copied, setCopied] = useState(false);
  const contentEndRef = useRef(null);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(solution);
      setCopied(true);
      toast.success('Solution copied to clipboard!');
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      toast.error('Failed to copy solution');
    }
  };

  const handleDownload = () => {
    const blob = new Blob([solution], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${storyId || 'solution'}_${new Date().toISOString().split('T')[0]}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    toast.success('Solution downloaded!');
  };

  useEffect(() => {
    if (solution) {
      setIsTyping(true);
      // For streaming, show content immediately
      if (isStreaming) {
        setDisplayedSolution(solution);
        // Auto-scroll to bottom as content streams (throttled)
        const scrollTimer = setTimeout(() => {
          if (contentEndRef.current) {
            contentEndRef.current.scrollIntoView({ 
              behavior: 'smooth', 
              block: 'end',
              inline: 'nearest'
            });
          }
        }, 600);
        return () => clearTimeout(scrollTimer);
      } else {
        // Animate final solution appearance if not streaming
        let currentIndex = 0;
        const interval = setInterval(() => {
          if (currentIndex <= solution.length) {
            setDisplayedSolution(solution.slice(0, currentIndex));
            currentIndex += 10; // Adjust speed
          } else {
            setIsTyping(false);
            clearInterval(interval);
          }
        }, 20);
        return () => clearInterval(interval);
      }
    }
  }, [solution, isStreaming]);

  const getProgressStage = () => {
    if (!progress) return null;
    const stages = {
      extracting_questions: { label: 'Analyzing Story', icon: '🔍', progress: 20 },
      questions_extracted: { label: 'Questions Extracted', icon: '✅', progress: 30 },
      analyzing_intent: { label: 'Understanding Intent', icon: '🧠', progress: 40 },
      searching_knowledge_base: { label: 'Searching Knowledge Base', icon: '📚', progress: 60 },
      retrieval_complete: { label: 'Retrieval Complete', icon: '📖', progress: 80 },
      generating_solution: { label: 'Generating Solution', icon: '✨', progress: 90 },
    };
    return stages[progress.step] || null;
  };

  const progressStage = getProgressStage();

  return (
    <Fade in={!!solution} timeout={800}>
      <Paper
        elevation={0}
        sx={{
          mt: 3,
          borderRadius: 3,
          overflow: 'hidden',
          background: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.05)} 0%, ${alpha(theme.palette.secondary?.main || theme.palette.primary.main, 0.05)} 100%)`,
          border: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
          position: 'relative',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            height: '4px',
            background: `linear-gradient(90deg, ${theme.palette.primary.main}, ${theme.palette.secondary?.main || theme.palette.primary.main})`,
            animation: isStreaming ? 'shimmer 2s infinite' : 'none',
            '@keyframes shimmer': {
              '0%': { transform: 'translateX(-100%)' },
              '100%': { transform: 'translateX(100%)' },
            },
          },
        }}
      >
        {/* Progress Indicator */}
        {isStreaming && progressStage && (
          <Box
            sx={{
              p: 2,
              background: alpha(theme.palette.primary.main, 0.08),
              borderBottom: `1px solid ${alpha(theme.palette.primary.main, 0.1)}`,
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 1 }}>
              <Box
                sx={{
                  width: 40,
                  height: 40,
                  borderRadius: '50%',
                  background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary?.main || theme.palette.primary.main})`,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: '1.5rem',
                  animation: 'pulse 2s infinite',
                  '@keyframes pulse': {
                    '0%, 100%': { transform: 'scale(1)' },
                    '50%': { transform: 'scale(1.1)' },
                  },
                }}
              >
                {progressStage.icon}
              </Box>
              <Box sx={{ flex: 1 }}>
                <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5 }}>
                  {progressStage.label}
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={progressStage.progress}
                  sx={{
                    height: 6,
                    borderRadius: 3,
                    backgroundColor: alpha(theme.palette.primary.main, 0.1),
                    '& .MuiLinearProgress-bar': {
                      borderRadius: 3,
                      background: `linear-gradient(90deg, ${theme.palette.primary.main}, ${theme.palette.secondary?.main || theme.palette.primary.main})`,
                    },
                  }}
                />
              </Box>
            </Box>
          </Box>
        )}

        {/* Header */}
        <Box sx={{ p: 3, pb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Box
                sx={{
                  width: 48,
                  height: 48,
                  borderRadius: 2,
                  background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary?.main || theme.palette.primary.main})`,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'white',
                  boxShadow: `0 4px 20px ${alpha(theme.palette.primary.main, 0.3)}`,
                }}
              >
                {isStreaming ? (
                  <SparkleIcon sx={{ animation: 'spin 2s linear infinite', '@keyframes spin': { '0%': { transform: 'rotate(0deg)' }, '100%': { transform: 'rotate(360deg)' } } }} />
                ) : (
                  <CheckCircleIcon />
                )}
              </Box>
              <Box>
                <Typography variant="h5" sx={{ fontWeight: 700, mb: 0.5 }}>
                  {isStreaming ? 'Generating Solution...' : 'Solution Generated'}
                </Typography>
                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                  {storyId && (
                    <Chip
                      label={storyId}
                      size="small"
                      sx={{
                        background: `linear-gradient(135deg, ${alpha(theme.palette.primary.main, 0.1)}, ${alpha(theme.palette.secondary?.main || theme.palette.primary.main, 0.1)})`,
                        border: `1px solid ${alpha(theme.palette.primary.main, 0.2)}`,
                        fontWeight: 600,
                      }}
                    />
                  )}
                  {processingTime && (
                    <Chip
                      label={`${processingTime.toFixed(2)}s`}
                      size="small"
                      icon={<RadioIcon sx={{ fontSize: 12 }} />}
                      sx={{
                        background: alpha(theme.palette.success.main, 0.1),
                        color: theme.palette.success.main,
                        fontWeight: 600,
                      }}
                    />
                  )}
                </Box>
              </Box>
            </Box>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <Tooltip title={copied ? 'Copied!' : 'Copy solution'}>
                <IconButton onClick={handleCopy} color="primary" size="small">
                  {copied ? <CheckCircleIcon /> : <ContentCopyIcon />}
                </IconButton>
              </Tooltip>
              <Tooltip title="Download as markdown">
                <IconButton onClick={handleDownload} color="primary" size="small">
                  <DownloadIcon />
                </IconButton>
              </Tooltip>
            </Box>
          </Box>
        </Box>

        {/* Streaming Content */}
        <Box
          sx={{
            p: 3,
            pt: 0,
            position: 'relative',
            maxHeight: 'calc(100vh - 300px)',
            overflowY: 'auto',
            '&::-webkit-scrollbar': {
              width: '8px',
            },
            '&::-webkit-scrollbar-track': {
              background: 'transparent',
            },
            '&::-webkit-scrollbar-thumb': {
              background: alpha(theme.palette.primary.main, 0.2),
              borderRadius: '4px',
              '&:hover': {
                background: alpha(theme.palette.primary.main, 0.3),
              },
            },
            '&::after': isStreaming ? {
              content: '"▋"',
              color: theme.palette.primary.main,
              animation: 'blink 1s infinite',
              '@keyframes blink': {
                '0%, 50%': { opacity: 1 },
                '51%, 100%': { opacity: 0 },
              },
            } : {},
          }}
        >
          <Box
            className="markdown-content"
            sx={{
              '& pre': {
                backgroundColor: theme.palette.mode === 'dark' ? '#1e1e1e' : '#f5f5f5',
                color: theme.palette.mode === 'dark' ? '#d4d4d4' : '#333',
                padding: '1.5em',
                borderRadius: '12px',
                overflow: 'auto',
                fontSize: '0.9em',
                border: `1px solid ${alpha(theme.palette.divider, 0.5)}`,
              },
              '& code': {
                backgroundColor: alpha(theme.palette.primary.main, 0.1),
                padding: '0.2em 0.5em',
                borderRadius: '6px',
                fontSize: '0.9em',
                fontFamily: 'monospace',
                color: theme.palette.primary.main,
                fontWeight: 500,
              },
              '& pre code': {
                backgroundColor: 'transparent',
                padding: 0,
                color: 'inherit',
              },
              '& h1, & h2, & h3': {
                fontWeight: 700,
                marginTop: '1.5em',
                marginBottom: '0.5em',
              },
              '& p': {
                lineHeight: 1.8,
                marginBottom: '1em',
              },
            }}
          >
            <ReactMarkdown
              components={{
                code({ node, inline, className, children, ...props }) {
                  const match = /language-(\w+)/.exec(className || '');
                  return !inline && match ? (
                    <SyntaxHighlighter
                      style={vscDarkPlus}
                      language={match[1]}
                      PreTag="div"
                      {...props}
                    >
                      {String(children).replace(/\n$/, '')}
                    </SyntaxHighlighter>
                  ) : (
                    <code className={className} {...props}>
                      {children}
                    </code>
                  );
                },
              }}
            >
              {displayedSolution || solution || ''}
            </ReactMarkdown>
            {/* Invisible element for auto-scroll */}
            <div ref={contentEndRef} style={{ height: 1, width: '100%' }} />
          </Box>
        </Box>
      </Paper>
    </Fade>
  );
};

export default StreamingSolutionDisplay;

