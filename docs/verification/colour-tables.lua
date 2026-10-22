--[[
Colours table cells by their content, for HTML and Typst alike.

  [text]{.good}  [text]{.fair}  [text]{.poor}   green, amber, red
  [0.85]{.score}                                a fraction recovered: red at 0, amber at 0.5, green at 1

Use it with `filters: [colour-tables.lua]` in a document's header.
]]

local GRADE = {good = '#c7e9c0', fair = '#fee6a8', poor = '#fcbba1'}

local function mix(a, b, t)
  local function ch(h, i) return tonumber(h:sub(i, i + 1), 16) end
  local out = '#'
  for i = 2, 6, 2 do out = out .. string.format('%02x', math.floor(ch(a, i) + (ch(b, i) - ch(a, i))*t + 0.5)) end
  return out
end

local function score_colour(x)
  x = math.max(0, math.min(1, x))
  if x < 0.5 then return mix(GRADE.poor, GRADE.fair, x/0.5) end
  return mix(GRADE.fair, GRADE.good, (x - 0.5)/0.5)
end

local function cell_colour(cell)
  local colour
  pandoc.Div(cell.contents):walk({Span = function(s)
    for name, c in pairs(GRADE) do
      if s.classes:includes(name) then colour = c end
    end
    if s.classes:includes('score') then
      local x = tonumber((pandoc.utils.stringify(s):gsub('[^%d%.%-]', '')))
      if x then colour = score_colour(x) end
    end
  end})
  return colour
end

local function paint(rows)
  for _, row in ipairs(rows) do
    for _, cell in ipairs(row.cells) do
      local c = cell_colour(cell)
      if c then
        if quarto.doc.is_format('typst') then
          cell.attributes['typst:fill'] = 'rgb("' .. c .. '")'
        else
          cell.attributes['style'] = 'background-color:' .. c
        end
      end
    end
  end
end

function Table(t)
  for _, body in ipairs(t.bodies) do paint(body.body) end
  return t
end
