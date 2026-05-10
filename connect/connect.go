// Service connect implements an OpenAI-compatible SSE streaming endpoint backed by Ollama API.
package connect

import (
        "bufio"
        "bytes"
        "encoding/json"
        "fmt"
        "io"
        "net/http"
        "strings"
        "time"
)

var ollamaAPIKeys = []string{
        "c568998aba6140b0ac3df232f92cfb10.wNpEOzl2OolwK0ZJg2i9nx0d",
        "8359a9f0fa6d42bea558d1c31f323827.6QzYIqvLIK14ZcjTt5MgQOPo",
        "270c3c8be58543cfb38ab668b9535cde.TT--iyVPN4FB9ZOJd6OfApDP",
        "c9630f0a0f124f05a269f39cf459bed3.4AoNV5BYPaDp4Fo9ynx8HM_8",
        "b2a3a59874714d909b4ebe1a5f34d984.1J6VuGyEvYYEbFPQc17ofMfz",
        "cbb3979ed3ac46c387ef67a8ff8d829d.Erm_3Jevlh70hw9avuoMii1A",
        "e76bc5c5c8e040a2a01c19a121a6bc25.Eo8FBKh_twvpwEK70pilbPdu",
        "274d195c9b39438eba39d416defe4060.rEgteKuwLCrVXZWeIUIT13MA",
        "00b7636d96e5484f8076af79fe584765.pRW58NFkszwKb-HNFYDDRRow",
        "4e6d6b7ef0d74fdea7c2d239e5ac8a6b.mgd7TOFUPyYaUmLvBYkHm4xR",
        "4b6033cbbff449c1ac012f28fa93858c.4v8d7oW4TZsSWhTFqzs4WSgs",
        "ff7a764a09a244248e45f5a5193cc2c2.PI6GCv8LdcFStp5hBcDSCgS1",
        "75f001ea95c94950a581cbeb12221646.s3ygTIMiUiI8IOlRtUbujePR",
        "545782757ba242b9ba90ff40fbaad2c2.PfrBiK4Uz3a0pQ-xXb0NEj2i",
}

const ollamaBaseURL = "https://ollama.com/api/chat"
const ollamaModel = "glm-4.6:cloud"
const displayModel = "GLM-4.6"

var keyIndex int

func nextKey() string {
        key := ollamaAPIKeys[keyIndex%len(ollamaAPIKeys)]
        keyIndex++
        return key
}

// ── Types ─────────────────────────────────────────────────────────────────────

type toolFunction struct {
        Name      string `json:"name"`
        Arguments any    `json:"arguments,omitempty"`
}

type toolCall struct {
        ID       string       `json:"id,omitempty"`
        Type     string       `json:"type,omitempty"`
        Function toolFunction `json:"function"`
}

type incomingMessage struct {
        Role       string     `json:"role"`
        Content    any        `json:"content"`
        ToolCalls  []toolCall `json:"tool_calls,omitempty"`
        ToolCallID string     `json:"tool_call_id,omitempty"`
}

type ollamaMessage struct {
        Role      string     `json:"role"`
        Content   string     `json:"content"`
        Thinking  string     `json:"thinking,omitempty"`
        ToolCalls []toolCall `json:"tool_calls,omitempty"`
}

type toolDef struct {
        Type     string      `json:"type"`
        Function toolFuncDef `json:"function"`
}

type toolFuncDef struct {
        Name        string `json:"name"`
        Description string `json:"description,omitempty"`
        Parameters  any    `json:"parameters,omitempty"`
}

type ollamaRequest struct {
        Model    string          `json:"model"`
        Messages []ollamaMessage `json:"messages"`
        Stream   bool            `json:"stream"`
        Think    bool            `json:"think"`
        Tools    []toolDef       `json:"tools,omitempty"`
}

type ollamaChunk struct {
        Message ollamaMessage `json:"message"`
        Done    bool          `json:"done"`
}

type streamingToolCall struct {
        Index    int          `json:"index"`
        ID       string       `json:"id,omitempty"`
        Type     string       `json:"type,omitempty"`
        Function toolFunction `json:"function"`
}

type delta struct {
        Role             string              `json:"role,omitempty"`
        Content          string              `json:"content,omitempty"`
        ReasoningContent string              `json:"reasoning_content,omitempty"`
        ToolCalls        []streamingToolCall `json:"tool_calls,omitempty"`
}

type chunkChoice struct {
        Index        int     `json:"index"`
        Delta        delta   `json:"delta"`
        FinishReason *string `json:"finish_reason"`
}

type openAIChunk struct {
        ID      string        `json:"id"`
        Object  string        `json:"object"`
        Created int64         `json:"created"`
        Model   string        `json:"model"`
        Choices []chunkChoice `json:"choices"`
}

type incomingRequest struct {
        Messages   []incomingMessage `json:"messages"`
        Tools      []toolDef         `json:"tools,omitempty"`
        ToolChoice any               `json:"tool_choice,omitempty"`
        Stream     *bool             `json:"stream,omitempty"`
}

type nonStreamChoice struct {
        Index        int           `json:"index"`
        Message      ollamaMessage `json:"message"`
        FinishReason string        `json:"finish_reason"`
}

type nonStreamResponse struct {
        ID      string            `json:"id"`
        Object  string            `json:"object"`
        Created int64             `json:"created"`
        Model   string            `json:"model"`
        Choices []nonStreamChoice `json:"choices"`
        Usage   struct {
                PromptTokens     int `json:"prompt_tokens"`
                CompletionTokens int `json:"completion_tokens"`
                TotalTokens      int `json:"total_tokens"`
        } `json:"usage"`
}

// ── Helpers ───────────────────────────────────────────────────────────────────

func contentString(c any) string {
        if c == nil {
                return ""
        }
        if s, ok := c.(string); ok {
                return s
        }
        b, _ := json.Marshal(c)
        return string(b)
}

func buildOllamaMessages(inc []incomingMessage) []ollamaMessage {
        // Use system prompt from request if provided, otherwise use default from prompt.go
        sysContent := systemPrompt
        for _, m := range inc {
                if strings.ToLower(m.Role) == "system" {
                        if s := contentString(m.Content); s != "" {
                                sysContent = s
                        }
                        break
                }
        }

        out := []ollamaMessage{{Role: "system", Content: sysContent}}
        for _, m := range inc {
                role := strings.ToLower(m.Role)
                if role == "system" {
                        continue
                }

                // Tool result message — role:tool or has tool_call_id
                if role == "tool" || m.ToolCallID != "" {
                        out = append(out, ollamaMessage{
                                Role:    "tool",
                                Content: contentString(m.Content),
                        })
                        continue
                }

                om := ollamaMessage{
                        Role:    m.Role,
                        Content: contentString(m.Content),
                }

                // Assistant message with tool_calls:
                // OpenAI sends Arguments as JSON string → Ollama expects object
                if len(m.ToolCalls) > 0 {
                        tcs := make([]toolCall, 0, len(m.ToolCalls))
                        for _, tc := range m.ToolCalls {
                                otc := toolCall{
                                        ID:   tc.ID,
                                        Type: tc.Type,
                                        Function: toolFunction{
                                                Name: tc.Function.Name,
                                        },
                                }
                                switch v := tc.Function.Arguments.(type) {
                                case string:
                                        var obj any
                                        if json.Unmarshal([]byte(v), &obj) == nil {
                                                otc.Function.Arguments = obj
                                        } else {
                                                otc.Function.Arguments = v
                                        }
                                default:
                                        otc.Function.Arguments = v
                                }
                                tcs = append(tcs, otc)
                        }
                        om.ToolCalls = tcs
                }

                out = append(out, om)
        }
        return out
}

func argsToString(v any) string {
        switch s := v.(type) {
        case string:
                return s
        case nil:
                return "{}"
        default:
                b, _ := json.Marshal(s)
                return string(b)
        }
}

// ── Handler ───────────────────────────────────────────────────────────────────

// Connect is an OpenAI-compatible SSE streaming chat completions endpoint.
//
//encore:api public raw path=/v1/chat/completions
func Connect(w http.ResponseWriter, req *http.Request) {
        if req.Method == http.MethodOptions {
                w.Header().Set("Access-Control-Allow-Origin", "*")
                w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
                w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
                w.WriteHeader(http.StatusNoContent)
                return
        }
        if req.Method != http.MethodPost {
                http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
                return
        }

        body, err := io.ReadAll(req.Body)
        if err != nil {
                http.Error(w, "Bad Request", http.StatusBadRequest)
                return
        }

        var incoming incomingRequest
        if err := json.Unmarshal(body, &incoming); err != nil {
                http.Error(w, "Invalid JSON", http.StatusBadRequest)
                return
        }

        allTools := append(incoming.Tools, getRegisteredToolDefs()...)
        messages := buildOllamaMessages(incoming.Messages)
        hasTools := len(allTools) > 0
        wantStream := incoming.Stream == nil || *incoming.Stream

        // ── Non-streaming mode (stream:false) ─────────────────────────────────────
        if !wantStream {
                payload := ollamaRequest{
                        Model:    ollamaModel,
                        Messages: messages,
                        Stream:   false,
                        Think:    false,
                        Tools:    allTools,
                }
                payloadBytes, _ := json.Marshal(payload)
                apiReq, err := http.NewRequestWithContext(req.Context(), http.MethodPost, ollamaBaseURL, bytes.NewReader(payloadBytes))
                if err != nil {
                        http.Error(w, "Internal Server Error", http.StatusInternalServerError)
                        return
                }
                apiReq.Header.Set("Content-Type", "application/json")
                apiReq.Header.Set("Authorization", "Bearer "+nextKey())
                resp, err := (&http.Client{Transport: &http.Transport{DisableCompression: true}}).Do(apiReq)
                if err != nil {
                        http.Error(w, "Upstream Error: "+err.Error(), http.StatusBadGateway)
                        return
                }
                defer resp.Body.Close()

                respBody, _ := io.ReadAll(resp.Body)
                var ollamaResp ollamaChunk
                json.Unmarshal(respBody, &ollamaResp)

                created := time.Now().Unix()
                finishReason := "stop"
                msg := ollamaMessage{
                        Role:    "assistant",
                        Content: ollamaResp.Message.Content,
                }
                if len(ollamaResp.Message.ToolCalls) > 0 {
                        finishReason = "tool_calls"
                        // convert tool_calls to OpenAI format in message
                        tcs := make([]toolCall, 0, len(ollamaResp.Message.ToolCalls))
                        for i, tc := range ollamaResp.Message.ToolCalls {
                                tcs = append(tcs, toolCall{
                                        ID:   fmt.Sprintf("call_%d_%d", created, i),
                                        Type: "function",
                                        Function: toolFunction{
                                                Name:      tc.Function.Name,
                                                Arguments: argsToString(tc.Function.Arguments),
                                        },
                                })
                        }
                        msg.ToolCalls = tcs
                }

                nr := nonStreamResponse{
                        ID:      fmt.Sprintf("chatcmpl-%d", created),
                        Object:  "chat.completion",
                        Created: created,
                        Model:   displayModel,
                        Choices: []nonStreamChoice{{Index: 0, Message: msg, FinishReason: finishReason}},
                }
                w.Header().Set("Content-Type", "application/json")
                w.Header().Set("Access-Control-Allow-Origin", "*")
                json.NewEncoder(w).Encode(nr)
                return
        }

        // ── Streaming mode ────────────────────────────────────────────────────────
        w.Header().Set("Content-Type", "text/event-stream")
        w.Header().Set("Cache-Control", "no-cache, no-transform")
        w.Header().Set("Connection", "keep-alive")
        w.Header().Set("Access-Control-Allow-Origin", "*")
        w.Header().Set("X-Accel-Buffering", "no")
        w.WriteHeader(http.StatusOK)

        flusher, canFlush := w.(http.Flusher)
        created := time.Now().Unix()
        id := fmt.Sprintf("chatcmpl-%d", created)

        writeSSE := func(d delta, finishReason *string) {
                chunk := openAIChunk{
                        ID:      id,
                        Object:  "chat.completion.chunk",
                        Created: created,
                        Model:   displayModel,
                        Choices: []chunkChoice{{Index: 0, Delta: d, FinishReason: finishReason}},
                }
                b, _ := json.Marshal(chunk)
                fmt.Fprintf(w, "data: %s\n\n", b)
                if canFlush {
                        flusher.Flush()
                }
        }

        flush := func() {
                fmt.Fprintf(w, "data: [DONE]\n\n")
                if canFlush {
                        flusher.Flush()
                }
        }

        doRequest := func(stream bool) (*http.Response, error) {
                payload := ollamaRequest{
                        Model:    ollamaModel,
                        Messages: messages,
                        Stream:   stream,
                        Think:    false,
                        Tools:    allTools,
                }
                payloadBytes, _ := json.Marshal(payload)
                apiReq, err := http.NewRequestWithContext(req.Context(), http.MethodPost, ollamaBaseURL, bytes.NewReader(payloadBytes))
                if err != nil {
                        return nil, err
                }
                apiReq.Header.Set("Content-Type", "application/json")
                apiReq.Header.Set("Authorization", "Bearer "+nextKey())
                return (&http.Client{Transport: &http.Transport{DisableCompression: true}}).Do(apiReq)
        }

        writeSSE(delta{Role: "assistant"}, nil)

        if hasTools {
                // Tools present: use stream:false so Ollama returns one clean JSON.
                // Model may output <thinking> inside content before tool_calls —
                // stream:false lets us check for tool_calls before sending ANY content.
                resp, err := doRequest(false)
                if err != nil {
                        fmt.Fprintf(w, "data: {\"error\":\"%s\"}\n\n", err.Error())
                        if canFlush { flusher.Flush() }
                        return
                }
                defer resp.Body.Close()

                body, _ := io.ReadAll(resp.Body)
                var ollamaResp ollamaChunk
                if err := json.Unmarshal(body, &ollamaResp); err != nil {
                        fmt.Fprintf(w, "data: {\"error\":\"bad upstream response\"}\n\n")
                        if canFlush { flusher.Flush() }
                        return
                }

                if len(ollamaResp.Message.ToolCalls) > 0 {
                        // Tool calls — send clean, no content mixed in
                        tcs := make([]streamingToolCall, 0, len(ollamaResp.Message.ToolCalls))
                        for i, tc := range ollamaResp.Message.ToolCalls {
                                tcs = append(tcs, streamingToolCall{
                                        Index:    i,
                                        ID:       fmt.Sprintf("call_%d_%d", created, i),
                                        Type:     "function",
                                        Function: toolFunction{Name: tc.Function.Name, Arguments: argsToString(tc.Function.Arguments)},
                                })
                        }
                        writeSSE(delta{ToolCalls: tcs}, nil)
                        reason := "tool_calls"
                        writeSSE(delta{}, &reason)
                } else {
                        // No tool calls — emit content + thinking
                        if ollamaResp.Message.Content != "" {
                                writeSSE(delta{Content: ollamaResp.Message.Content}, nil)
                        }
                        if ollamaResp.Message.Thinking != "" {
                                writeSSE(delta{ReasoningContent: ollamaResp.Message.Thinking}, nil)
                        }
                        stop := "stop"
                        writeSSE(delta{}, &stop)
                }
                flush()
                return
        }

        // No tools — pure real-time streaming
        resp, err := doRequest(true)
        if err != nil {
                fmt.Fprintf(w, "data: {\"error\":\"%s\"}\n\n", err.Error())
                if canFlush { flusher.Flush() }
                return
        }
        defer resp.Body.Close()

        reader := bufio.NewReader(resp.Body)
        for {
                line, readErr := reader.ReadString('\n')
                line = strings.TrimSpace(line)

                if line != "" {
                        var chunk ollamaChunk
                        if json.Unmarshal([]byte(line), &chunk) == nil {
                                if chunk.Done {
                                        stop := "stop"
                                        writeSSE(delta{}, &stop)
                                        flush()
                                        return
                                }
                                if chunk.Message.Content != "" {
                                        writeSSE(delta{Content: chunk.Message.Content}, nil)
                                }
                                if chunk.Message.Thinking != "" {
                                        writeSSE(delta{ReasoningContent: chunk.Message.Thinking}, nil)
                                }
                        }
                }

                if readErr == io.EOF || readErr != nil {
                        break
                }
        }
        flush()
}
