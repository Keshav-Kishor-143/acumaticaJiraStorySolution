import React from 'react';
import {
  Box,
  Paper,
  Typography,
  Chip,
  Divider,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  ContentCopy as ContentCopyIcon,
  Download as DownloadIcon,
  CheckCircle as CheckCircleIcon,
} from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import toast from 'react-hot-toast';

const SolutionDisplay = ({ solution, storyId, processingTime, savedFilePath }) => {
  const [copied, setCopied] = React.useState(false);

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

  if (!solution) {
    return null;
  }

  return (
    <Paper
      elevation={2}
      sx={{
        p: 3,
        borderRadius: 2,
        mt: 3,
      }}
    >
      {/* Header */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
        <Box>
          <Typography variant="h5" sx={{ fontWeight: 600, mb: 1 }}>
            Generated Solution
          </Typography>
          {storyId && (
            <Chip
              label={storyId}
              color="primary"
              size="small"
              sx={{ mr: 1 }}
            />
          )}
          {processingTime && (
            <Chip
              label={`Processed in ${processingTime.toFixed(2)}s`}
              size="small"
              variant="outlined"
            />
          )}
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

      {savedFilePath && (
        <Box sx={{ mb: 2 }}>
          <Chip
            label={`Saved to: ${savedFilePath}`}
            size="small"
            color="success"
            variant="outlined"
          />
        </Box>
      )}

      <Divider sx={{ mb: 3 }} />

      {/* Markdown Content */}
      <Box
        className="markdown-content"
        sx={{
          '& pre': {
            backgroundColor: '#1e1e1e',
            color: '#d4d4d4',
            padding: '1em',
            borderRadius: '8px',
            overflow: 'auto',
            fontSize: '0.9em',
          },
          '& code': {
            backgroundColor: '#f5f5f5',
            padding: '0.2em 0.4em',
            borderRadius: '4px',
            fontSize: '0.9em',
            fontFamily: 'monospace',
          },
          '& pre code': {
            backgroundColor: 'transparent',
            padding: 0,
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
          {solution}
        </ReactMarkdown>
      </Box>
    </Paper>
  );
};

export default SolutionDisplay;

